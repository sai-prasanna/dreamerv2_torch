import functools
import re
from typing import List, Dict, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.distributions as td
import torch.functional as F

import common
from .dists import (
    Bernoulli,
    OneHotDist,
    ContDist,
    TanhBijector,
    SampleDist,
    TruncNormalDist,
)

# Represents TD model in PlaNET
class EnsembleRSSM(nn.Module):
    """RSSM is the core recurrent part of the PlaNET module. It consists of
    two networks, one (obs) to calculate posterior beliefs and states and
    the second (img) to calculate prior beliefs and states. The prior network
    takes in the previous state and action, while the posterior network takes
    in the previous state, action, and a latent embedding of the most recent
    observation.
    """

    def __init__(
        self,
        action_size: int,
        embed_size: int,
        ensemble: int = 5,
        stoch: int = 30,
        deter: int = 200,
        hidden: int = 200,
        discrete: int = 0,
        act: str = "ELU",
        std_act: str = "softplus",
        min_std: float = 0.1,
        norm="none",
    ):
        super().__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        if self._discrete:
            imag_input_dim = self._discrete * self._stoch + action_size
            self.state_dim = self._discrete * self._stoch + self._deter
        else:
            imag_input_dim = self._stoch + action_size
            self.state_dim = self._stoch + self._deter
        self._act = get_act(act)
        obs_out_layer = [nn.Linear(embed_size + deter, hidden)]
        if norm == "layer":
            obs_out_layer.append(nn.LayerNorm(hidden))
        obs_out_layer.append(self._act())
        self.obs_out_layer = nn.Sequential(*obs_out_layer)

        if self._discrete > 0:
            self.obs_suff_stats = nn.Linear(hidden, stoch * discrete)
        else:
            self.obs_suff_stats = nn.Linear(hidden, 2 * stoch)

        self.cell = GRUCell(inp_size=self._hidden, out_size=self._deter, norm=True)
        img_inp = [nn.Linear(imag_input_dim, hidden)]
        if norm == "layer":
            img_inp.append(nn.LayerNorm(hidden))
        img_inp.append(self._act())

        self.img_inp_layer = nn.Sequential(*img_inp)

        img_suff_stats_ensemble = []
        for i in range(ensemble):
            img_layers = []
            img_layers.append(nn.Linear(deter, hidden))
            if norm == "layer":
                img_layers.append(nn.LayerNorm(hidden))
            img_layers.append(self._act())
            if self._discrete > 0:
                img_layers.append(nn.Linear(hidden, stoch * discrete))
            else:
                img_layers.append(nn.Linear(hidden, 2 * stoch))
            img_suff_stats_ensemble.append(nn.Sequential(*img_layers))
        self.img_suff_stats_ensemble = nn.ModuleList(img_suff_stats_ensemble)

        self.std_act = std_act
        self.min_std = min_std

    def initial(self, batch_size: int) -> Dict[str, Tensor]:
        """Returns the inital state for the RSSM, which consists of mean,
        std for the stochastic state, the sampled stochastic hidden state
        (from mean, std), and the deterministic hidden state, which is
        pushed through the GRUCell.

        Args:
            batch_size (int): Batch size for initial state

        Returns:
            List of tensors
        """
        device = next(self.parameters()).device
        if self._discrete:
            state = {
                "logit": torch.zeros(
                    batch_size, self._stoch, self._discrete, device=device
                ),
                "stoch": torch.zeros(
                    batch_size, self._stoch, self._discrete, device=device
                ),
                "deter": torch.zeros(batch_size, self._deter, device=device),
            }
        else:
            state = {
                "mean": torch.zeros(batch_size, self._stoch, device=device),
                "std": torch.zeros(batch_size, self._stoch, device=device),
                "stoch": torch.zeros(batch_size, self._stoch, device=device),
                "deter": torch.zeros(batch_size, self._deter, device=device),
            }
        return state

    def observe(
        self,
        embed: Tensor,
        action: Tensor,
        is_first: Tensor,
        state: List[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Returns the corresponding states from the embedding from ConvEncoder
        and actions. This is accomplished by rolling out the RNN from the
        starting state through each index of embed and action, saving all
        intermediate states between.

        Args:
            embed (Tensor): ConvEncoder embedding
            action (Tensor): Actions
            state (List[Tensor]): Initial state before rollout

        Returns:
            Posterior states and prior states (both List[Tensor])
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.size()[0])

        embed, action, is_first = (
            swap(embed),
            swap(action),
            swap(is_first),
        )  # T x B x enc_dim, T x B x action_dim

        post, prior = common.static_scan(
            lambda prev, *inputs: self.obs_step(prev[0], *inputs),
            (action, embed, is_first),
            (state, state),
        )
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action: Tensor, state: List[Tensor] = None) -> List[Tensor]:
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = common.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state: Dict[str, Tensor]) -> Tensor:
        # Constructs feature for input to reward, decoder, actor, critic
        stoch = state["stoch"]
        if self._discrete:
            shape = stoch.shape[:-2] + (self._stoch * self._discrete,)
            stoch = stoch.reshape(shape)
        return torch.concat([stoch, state["deter"]], -1)

    def get_dist(self, state: Dict[str, Tensor], ensemble=False) -> Tensor:
        if ensemble:
            state = self._suff_stats_ensemble(state["deter"])
        if self._discrete:
            logit = state["logit"]
            dist = td.Independent(OneHotDist(logits=logit), 1)
        else:
            mean, std = state["mean"], state["std"]
            dist = ContDist(td.Independent(td.Normal(mean, std), 1))
        return dist

    def obs_step(
        self,
        prev_state: Dict[str, Tensor],
        prev_action: Tensor,
        embed: Tensor,
        is_first: Tensor,
        sample=True,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Runs through the posterior model and returns the posterior state

        Args:
            prev_state (Tensor): The previous state
            prev_action (Tensor): The previous action
            embed (Tensor): Embedding from ConvEncoder

        Returns:
            Post and Prior state
        """
        for k, v in prev_state.items():
            prev_state[k] = torch.einsum(
                "b,b...->b...", 1.0 - is_first.type(v.dtype), v
            )
        prev_action = torch.einsum(
            "b,b...->b...", 1.0 - is_first.type(prev_action.dtype), prev_action
        )
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], dim=-1)
        x = self.obs_out_layer(x)
        stats = self._suff_stats(x, self.obs_suff_stats)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(
        self, prev_state: Dict[str, Tensor], prev_action: Tensor, sample=True
    ) -> List[Tensor]:
        """Runs through the prior model and returns the prior state

        Args:
            prev_state (Tensor): The previous state
            prev_action (Tensor): The previous action

        Returns:
            Prior state
        """
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.reshape(shape)
        x = torch.cat([prev_stoch, prev_action], dim=-1)
        x = self.img_inp_layer(x)
        deter = self.cell(x, prev_state["deter"])
        x = deter
        stats = self._suff_stats_ensemble(x)
        index = (
            torch.randint(len(self.img_suff_stats_ensemble))
            if len(self.img_suff_stats_ensemble) > 1
            else 0
        )
        # Pick one randomly
        stats = {k: v[index] for k, v in stats.items()}
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def _suff_stats_ensemble(self, inp):
        # TODO: Optimize with torch vmap
        stats = [self._suff_stats(inp, m) for m in self.img_suff_stats_ensemble]
        # We are relying on the fact that stats[0].keys() is ordered the same always
        # True in python 3.8 and above I guess
        stats = {k: torch.stack([x[k] for x in stats]) for k in stats[0].keys()}
        return stats

    def _suff_stats(self, x, suff_stats_module):
        x = suff_stats_module(x)
        if self._discrete > 0:
            stats = {"logit": x.view(x.shape[:-1] + (self._stoch, self._discrete))}
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = {
                "softplus": lambda: F.Softplus()(std),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self.std_act]()
            std = std + self.min_std
            stats = {"mean": mean, "std": std}
        return stats

    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        kld = td.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        if balance == 0.5:
            value = kld(
                dist(lhs) if self._discrete else dist(lhs)._dist,
                dist(rhs) if self._discrete else dist(rhs)._dist,
            )
            loss = torch.mean(torch.maximum(value, free))
        else:
            value_lhs = value = kld(
                dist(lhs) if self._discrete else dist(lhs)._dist,
                dist(sg(rhs)) if self._discrete else dist(sg(rhs))._dist,
            )
            value_rhs = kld(
                dist(sg(lhs)) if self._discrete else dist(sg(lhs))._dist,
                dist(rhs) if self._discrete else dist(rhs)._dist,
            )
            if free_avg:
                loss_lhs = torch.maximum(torch.mean(value_lhs), torch.Tensor([free])[0])
                loss_rhs = torch.maximum(torch.mean(value_rhs), torch.Tensor([free])[0])
            else:
                loss_lhs = torch.maximum(value_lhs, torch.Tensor([free])[0]).mean()
                loss_rhs = torch.maximum(value_rhs, torch.Tensor([free])[0]).mean()

            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value


class Encoder(nn.Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="ELU",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=[400, 400, 400, 400],
    ):
        super().__init__()
        self.shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]
        print("Encoder CNN inputs:", list(self.cnn_keys))
        print("Encoder MLP inputs:", list(self.mlp_keys))
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers
        self.embed_size = 0
        if self.cnn_keys:
            channels = sum([self.shapes[k][-1] for k in self.cnn_keys])
            self.cnn_encoder = ConvEncoder(channels, cnn_depth, act, cnn_kernels, norm)
            cnn_shapes = [self.shapes[k] for k in self.cnn_keys]
            channels = sum([s[-1] for s in cnn_shapes])
            # Assuming height and width of all our multi-view image observations
            # is the same. Can later alter to have a different CNN and fuse

            height, width = self.shapes[self.cnn_keys[0]][0:2]
            self.cnn_output_shapes = self.cnn_encoder.compute_conv_output_shapes(
                (channels, height, width)
            )
            self.embed_size += np.prod(self.cnn_output_shapes[-1]) * len(self.cnn_keys)
        else:
            self.cnn_output_shapes = []
        if self.mlp_keys:
            state_dim = sum([self._shapes[k][-1] for k in self.mlp_keys])
            all_mlp_layers = [state_dim] + mlp_layers
            self.mlp_encoder = MLP(all_mlp_layers, act, norm)
            self.embed_size += all_mlp_layers[-1] * len(self.mlp_keys)

    def forward(self, data):
        outputs = []
        if self.cnn_keys:
            cnn_input = torch.cat(
                [v for k, v in data.items() if k in self.cnn_keys], dim=-1
            )
            outputs.append(self.cnn_encoder(cnn_input))
        if self.mlp_keys:
            mlp_input = torch.cat(
                [v for k, v in data.items() if k in self.mlp_keys], dim=-1
            )
            outputs.append(self.mlp_encoder(mlp_input))
        out = torch.cat(outputs, dim=-1)
        return out


class ConvEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        act: str,
        kernels=(4, 4, 4, 4),
        norm: str = "none",
    ):
        super(ConvEncoder, self).__init__()
        self._act = get_act(act)
        self._depth = depth
        self._kernels = kernels

        layers = []
        for i, kernel in enumerate(self._kernels):
            if i == 0:
                inp_dim = channels
            else:
                inp_dim = 2 ** (i - 1) * self._depth
            depth = 2**i * self._depth
            layers.append(nn.Conv2d(inp_dim, depth, kernel, stride=2))
            print(inp_dim, depth, kernel)
            layers.append(self._act())
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        x = x.permute(0, 3, 1, 2)

        x = self.layers(x)

        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        shape = list(obs.shape[:-3]) + [x.shape[-1]]
        return x.reshape(shape)

    def compute_conv_output_shapes(self, image_size):
        with torch.no_grad():
            x = torch.rand((1, *image_size)).to(self.layers[0].weight.device)
            # x = x.permute(0, 2,)
            shapes = [x.shape[1:]]
            for l in self.layers:
                x = l(x)
                if not type(l) != self._act:
                    shapes.append(x.shape[1:])
        return shapes


# class Decoder(nn.Module):
#     def __init__(
#         self,
#         shapes: Dict[str, List[List[int]]],
#         latent_dim: int,
#         cnn_enc_shapes: List[List[int]],
#         enc_conv_kernels: Sequence[int],
#         mlp_hidden_layers: Sequence[int] = (400, 400, 400, 400, 400),
#         act: str = "ELU",
#         norm: str = "none",
#         cnn_keys: str = r".*",
#         mlp_keys: str = r".*",
#     ):
#         super(Decoder, self).__init__()
#         self._shapes = shapes
#         self._act = act
#         self.cnn_keys = [
#             k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
#         ]
#         self.mlp_keys = [
#             k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
#         ]
#         self.cnn_enc_shapes = cnn_enc_shapes

#         if self.cnn_keys:
#             self.cnn_decoder = TConvDecoder(
#                 latent_dim, cnn_enc_shapes, enc_conv_kernels, act
#             )
#         if self.mlp_keys:
#             state_dim = sum([self._shapes[k][-1] for k in self.mlp_keys])
#             mlp_hidden_layers = [latent_dim] + list(mlp_hidden_layers) + [state_dim]
#             self.mlp_decoder = MLP(mlp_hidden_layers, act, norm)

#     def forward(self, features):
#         dists = {}
#         if self.cnn_keys:
#             cnn_outputs = [self._shapes[k][-1] for k in self.cnn_keys]
#             cnn_out = self.cnn_decoder(features)
#             means = torch.split(cnn_out, cnn_outputs, -1)
#             dists.update(
#                 {
#                     key: ContDist(td.Independent(td.Normal(mean, 1), 3))
#                     for key, mean in zip(self.cnn_keys, means)
#                 }
#             )
#         if self.mlp_keys:
#             mlp_outputs = [self._shapes[k][0] for k in self.mlp_keys]
#             mlp_out = self.mlp_decoder(features)
#             means = torch.split(mlp_out, mlp_outputs, -1)
#             dists.update(
#                 {
#                     key: ContDist(td.Independent(td.Normal(mean, 1), 1))  # MSE
#                     for key, mean in zip(self.mlp_keys, means)
#                 }
#             )
#         return dists


class Decoder(nn.Module):
    def __init__(
        self,
        shapes: Dict[str, List[List[int]]],
        latent_dim: int,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="ELU",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=[400, 400, 400, 400],
    ):
        super(Decoder, self).__init__()
        self._shapes = shapes
        self._act = act
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]

        if self.cnn_keys:
            out_channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
            self.cnn_decoder = ConvDecoder(latent_dim, cnn_depth, act, shape=(sum(out_channels.values()), 64, 64), kernels=cnn_kernels)
        if self.mlp_keys:
            state_dim = sum([self._shapes[k][-1] for k in self.mlp_keys])
            mlp_hidden_layers = [latent_dim] + list(mlp_layers) + [state_dim]
            self.mlp_decoder = MLP(mlp_hidden_layers, act, norm)

    def forward(self, features):
        dists = {}
        if self.cnn_keys:
            cnn_outputs = [self._shapes[k][-1] for k in self.cnn_keys]
            cnn_out = self.cnn_decoder(features)
            means = torch.split(cnn_out, cnn_outputs, -1)
            dists.update(
                {
                    key: ContDist(td.Independent(td.Normal(mean, 1), 3))
                    for key, mean in zip(self.cnn_keys, means)
                }
            )
        if self.mlp_keys:
            mlp_outputs = [self._shapes[k][0] for k in self.mlp_keys]
            mlp_out = self.mlp_decoder(features)
            means = torch.split(mlp_out, mlp_outputs, -1)
            dists.update(
                {
                    key: ContDist(td.Independent(td.Normal(mean, 1), 1))  # MSE
                    for key, mean in zip(self.mlp_keys, means)
                }
            )
        return dists


class ConvDecoder(nn.Module):
    def __init__(
        self,
        inp_depth,
        depth=32,
        act='ELU',
        shape=(3, 64, 64),
        kernels=(5, 5, 6, 6),
    ):
        super(ConvDecoder, self).__init__()
        self._inp_depth = inp_depth
        self._act = get_act(act)
        self._depth = depth
        self._shape = shape
        self._kernels = kernels

        self._linear_layer = nn.Linear(inp_depth, 32 * self._depth)
        inp_dim = 32 * self._depth

        cnnt_layers = []
        for i, kernel in enumerate(self._kernels):
            depth = 2 ** (len(self._kernels) - i - 2) * self._depth
            act = self._act
            if i == len(self._kernels) - 1:
                # depth = self._shape[-1]
                depth = self._shape[0]
                act = None
            if i != 0:
                inp_dim = 2 ** (len(self._kernels) - (i - 1) - 2) * self._depth
            cnnt_layers.append(nn.ConvTranspose2d(inp_dim, depth, kernel, 2))
            print((inp_dim, depth, kernel, 2))
            if act is not None:
                cnnt_layers.append(act())
        self._cnnt_layers = nn.Sequential(*cnnt_layers)

    def forward(self, features):
        x = self._linear_layer(features)
        x = x.reshape([-1, 1, 1, 32 * self._depth])
        x = x.permute(0, 3, 1, 2)
        x = self._cnnt_layers(x)
        mean = x.reshape(features.shape[:-1] + self._shape)
        mean = mean.permute(0, 1, 3, 4, 2)
        return mean


class TConvDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        enc_conv_output_sizes: List[List[int]],
        enc_conv_kernels: Sequence[int],
        act: str = "none",
    ):
        super(TConvDecoder, self).__init__()
        self._act = get_act(act)
        self._linear_layer = nn.Linear(
            latent_dim, functools.reduce(lambda a, b: a * b, enc_conv_output_sizes[-1])
        )
        self._tconv_output_shapes = list(reversed(enc_conv_output_sizes))
        self._tconv_kernels = list(reversed(enc_conv_kernels))
        layers = []
        for i in range(len(self._tconv_output_shapes) - 1):
            layers.append(
                nn.ConvTranspose2d(
                    self._tconv_output_shapes[i][0],
                    self._tconv_output_shapes[i + 1][0],
                    self._tconv_kernels[i],
                    2,
                )
            )
            print(self._tconv_output_shapes[i][0],
                    self._tconv_output_shapes[i + 1][0],
                    self._tconv_kernels[i],
                    2,)
            if i != len(self._tconv_output_shapes) - 2:
                layers.append(self._act())
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        x = self._linear_layer(features)
        x = x.view([-1, *self._tconv_output_shapes[0]])

        output_idx = 1
        for l in self.layers:
            if isinstance(l, nn.ConvTranspose2d):
                x = l(x, output_size=self._tconv_output_shapes[output_idx][1:])
                output_idx += 1
            else:
                x = l(x)
        mean = x.view(features.shape[:-1] + self._tconv_output_shapes[-1])
        return mean.permute(0, 1, 3, 4, 2)


class MLP(nn.Module):
    def __init__(self, layers, act: str, norm: str):
        super(MLP, self).__init__()
        act = get_act(act)
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if norm == "layer":
                modules.append(nn.LayerNorm(layers[i + 1]))
            modules.append(act())
        self.net = nn.Sequential(*modules)

    def forward(self, X):
        return self.net(X)


class MLPDistribution(nn.Module):
    def __init__(
        self,
        inp_size: int,
        out_size: int,
        layers: int,
        units: int,
        act: str = "ELU",
        norm: str = "none",
        **dist_layer,
    ):
        super(MLPDistribution, self).__init__()
        act = get_act(act)
        mlp_layers = [inp_size] + [units] * layers
        modules = []
        for i in range(len(mlp_layers) - 1):
            modules.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))
            if norm == "layer":
                modules.append(nn.LayerNorm(mlp_layers[i + 1]))
            modules.append(act())
        self.net = nn.Sequential(*modules)
        self.dist_layer = DistLayer(
            inp_size=mlp_layers[-1], out_size=out_size, **dist_layer
        )

    def forward(self, features):
        return self.dist_layer(self.net(features))


class GRUCell(nn.Module):
    def __init__(
        self,
        inp_size: int,
        out_size: int,
        norm: bool = False,
        act: str = "tanh",
        update_bias: int = -1,
    ):
        super().__init__()
        self._inp_size = inp_size
        self._out_size = out_size
        self._act = get_act(act)
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + out_size, 3 * out_size, bias=norm)
        if norm:
            self._norm = nn.LayerNorm(3 * out_size)

    @property
    def state_size(self):
        return self._out_size

    def forward(self, inputs, state):
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._out_size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output


class DistLayer(nn.Module):
    def __init__(
        self,
        inp_size: int,
        out_size: int,
        act: str = "ELU",
        dist: str = "mse",
        init_std: float = 0.0,
        min_std: float = 0.1,
        action_disc: int = 5,
        temp: float = 0.1,
        outscale: int = 0,
    ):
        super(DistLayer, self).__init__()
        self._dist = dist
        self._act = get_act(act)
        self._min_std = min_std
        self._init_std = init_std
        self._action_disc = action_disc
        self._temp = temp
        self._outscale = outscale
        self._out_size = out_size

        if self._dist in ["tanh_normal", "normal", "trunc_normal", "tanh_normal_5"]:
            self._dist_layer = nn.Linear(inp_size, 2 * out_size)
        elif self._dist in ["mse", "onehot", "onehot_gumble"]:
            self._dist_layer = nn.Linear(inp_size, out_size)
        else:
            raise NotImplementedError(self._dist)

    def forward(self, features):
        x = features
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._out_size] * 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = td.Normal(mean, std)
            dist = td.transformed_distribution.TransformedDistribution(
                dist, TanhBijector()
            )
            dist = SampleDist(td.Independent(dist, 1))
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._out_size] * 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = td.Normal(mean, std)
            dist = td.transformed_distribution.TransformedDistribution(
                dist, TanhBijector()
            )
            dist = SampleDist(td.Independent(dist, 1))
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._out_size] * 2, -1)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = td.Normal(mean, std)
            dist = ContDist(td.Independent(dist, 1))
        elif self._dist == "mse":
            mean = self._dist_layer(x)
            dist = td.Normal(mean, 1.0)
            dist = ContDist(td.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._out_size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = TruncNormalDist(mean, std, -1, 1)
            dist = ContDist(td.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = OneHotDist(x)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = ContDist(td.gumbel.Gumbel(x, 1 / temp))
        elif self._dist == "binary":
            return Bernoulli(
                td.independent.Independent(
                    td.bernoulli.Bernoulli(logits=mean), len(self._shape)
                )
            )
        else:
            raise NotImplementedError(self._dist)
        return dist


def get_act(name):
    if name == "none":
        return lambda x: x
    if name == "mish":
        return lambda x: x * torch.tanh(torch.nn.softplus(x))
    elif hasattr(torch.nn, name):
        return getattr(torch.nn, name)
    elif hasattr(torch, name):
        return getattr(torch, name)
    else:
        raise NotImplementedError(name)
