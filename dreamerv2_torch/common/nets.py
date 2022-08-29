import re

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.functional as F

from .other import static_scan
from .dists import *
from .torch_utils import Module


class EnsembleRSSM(Module):
    def __init__(
        self,
        ensemble=5,
        stoch=30,
        deter=200,
        hidden=200,
        discrete=False,
        act="elu",
        norm="none",
        std_act="softplus",
        min_std=0.1,
    ):
        super().__init__()
        self._ensemble = ensemble
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = get_act(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std
        self._cell = GRUCell(self._deter, norm=True)
        self._dummy = nn.Parameter(torch.ones(()))

    def initial(self, batch_size):
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

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: torch.permute(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        post, prior = static_scan(
            lambda prev, *inputs: self.obs_step(prev[0], *inputs),
            (swap(action), swap(embed), swap(is_first)),
            (state, state),
        )
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        swap = lambda x: torch.permute(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = swap(action)
        prior = static_scan(self.img_step, [action], state)[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = stoch.shape[:-2] + (self._stoch * self._discrete,)
            stoch = stoch.reshape(shape)
        return torch.concat([stoch, state["deter"]], -1)

    def get_dist(self, state, ensemble=False):
        if ensemble:
            state = self._suff_stats_ensemble(state["deter"])
        if self._discrete:
            logit = state["logit"]
            dist = td.Independent(OneHotDist(logits=logit), 1)
        else:
            mean, std = state["mean"], state["std"]
            dist = ContDist(td.Independent(td.Normal(mean, std), 1))
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # if is_first.any():
        prev_action = torch.einsum(
            "b,b...->b...", 1.0 - is_first.type(prev_action.dtype), prev_action
        )
        prev_state = {
            k: torch.einsum("b,b...->b...", 1.0 - is_first.type(v.dtype), v)
            for k, v in prev_state.items()
        }
        prior = self.img_step(prev_state, prev_action, sample)
        x = torch.concat([prior["deter"], embed], -1)
        x = self.get("obs_out", nn.Linear, x.shape[-1], self._hidden)(x)
        x = self.get("obs_out_norm", NormLayer, x.shape[-1], self._norm)(x)
        x = self._act(x)
        stats = self._suff_stats_layer("obs_dist", x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = prev_stoch.shape[:-2] + (self._stoch * self._discrete,)
            prev_stoch = torch.reshape(prev_stoch, shape)
        x = torch.concat([prev_stoch, prev_action], -1)
        x = self.get("img_in", nn.Linear, x.shape[-1], self._hidden)(x)
        x = self.get("img_in_norm", NormLayer, x.shape[-1], self._norm)(x)
        x = self._act(x)
        deter = self._cell(x, prev_state["deter"])
        x = deter
        stats = self._suff_stats_ensemble(x)
        index = torch.randint(self._ensemble) if self._ensemble > 1 else 0
        # Pick one randomly
        stats = {k: v[index] for k, v in stats.items()}
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def _suff_stats_ensemble(self, inp):
        bs = list(inp.shape[:-1])
        inp = inp.reshape([-1, inp.shape[-1]])
        stats = []
        for k in range(self._ensemble):
            x = self.get(f"img_out_{k}", nn.Linear, inp.shape[-1], self._hidden)(inp)
            x = self.get(f"img_out_norm_{k}", NormLayer, x.shape[-1], self._norm)(x)
            x = self._act(x)
            stats.append(self._suff_stats_layer(f"img_dist_{k}", x))
        stats = {k: torch.stack([x[k] for x in stats], 0) for k, v in stats[0].items()}
        stats = {
            k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
            for k, v in stats.items()
        }
        return stats

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(
                name, nn.Linear, x.shape[-1], self._stoch * self._discrete, None
            )(x)
            logit = torch.reshape(
                x,
                x.shape[:-1]
                + (
                    self._stoch,
                    self._discrete,
                ),
            )
            return {"logit": logit}
        else:
            x = self.get(name, nn.Linear, x.shape[-1], 2 * self._stoch, None)(x)
            mean, std = torch.chunk(x, 2, -1)
            std = {
                "softplus": lambda: nn.Softplus()(std),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
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


class Encoder(Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="elu",
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
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

    def forward(self, data):
        key, shape = list(self.shapes.items())[0]
        batch_dims = data[key].shape[: -len(shape)]
        data = {
            k: torch.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims) :])
            for k, v in data.items()
        }
        outputs = []
        if self.cnn_keys:
            outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
        if self.mlp_keys:
            outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
        output = torch.concat(outputs, -1)
        return output.reshape(batch_dims + output.shape[1:])

    def _cnn(self, data):
        x = torch.concat(list(data.values()), -1)
        x = x.permute(0, 3, 1, 2)
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2**i * self._cnn_depth
            x = self.get(f"conv{i}", nn.Conv2d, x.shape[-3], depth, kernel, 2)(x)
            x = self.get(f"convnorm{i}", NormLayer, x.shape[-3:], self._norm)(x)
            x = self._act(x)

        return x.reshape(tuple(x.shape[:-3]) + (-1,))

    def _mlp(self, data):
        x = torch.concat(list(data.values()), -1)
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f"dense{i}", nn.Dense, x.shape[-1], width)(x)
            x = self.get(f"densenorm{i}", NormLayer, x.shape[-1], self._norm)(x)
            x = self._act(x)
        return x


class Decoder(Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="elu",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=[400, 400, 400, 400],
    ):
        super().__init__()
        self._shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]
        print("Decoder CNN outputs:", list(self.cnn_keys))
        print("Decoder MLP outputs:", list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

    def forward(self, features):
        outputs = {}
        if self.cnn_keys:
            outputs.update(self._cnn(features))
        if self.mlp_keys:
            outputs.update(self._mlp(features))
        return outputs

    def _cnn(self, features):
        channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
        x = self.get("convin", nn.Linear, features.shape[-1], 32 * self._cnn_depth)(
            features
        )
        x = torch.reshape(x, [-1, 32 * self._cnn_depth, 1, 1])
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
            act, norm = self._act, self._norm
            if i == len(self._cnn_kernels) - 1:
                depth, act, norm = sum(channels.values()), lambda x: x, "none"
            x = self.get(f"conv{i}", nn.ConvTranspose2d, x.shape[-3], depth, kernel, 2)(
                x
            )
            x = self.get(f"convnorm{i}", NormLayer, x.shape[-3:], norm)(x)
            x = act(x)
        x = x.permute((0, 2, 3, 1))
        x = x.reshape(features.shape[:-1] + x.shape[1:])
        means = torch.split(x, list(channels.values()), -1)
        dists = {
            key: ContDist(td.Independent(td.Normal(mean, 1), 3))
            for (key, shape), mean in zip(channels.items(), means)
        }
        return dists

    def _mlp(self, features):
        shapes = {k: self._shapes[k] for k in self.mlp_keys}
        x = features
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f"dense{i}", nn.Linear, x.shape[-1], width)(x)
            x = self.get(f"densenorm{i}", NormLayer, x.shape[-1], self._norm)(x)
            x = self._act(x)
        dists = {}
        for key, shape in shapes.items():
            dists[key] = self.get(f"dense_{key}", DistLayer, shape)(x)
        return dists


class MLP(Module):
    def __init__(self, shape, layers, units, act="elu", norm="none", **out):
        super().__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self._out = out

    def forward(self, features):
        x = features
        x = x.reshape([-1, x.shape[-1]])
        for index in range(self._layers):
            x = self.get(f"dense{index}", nn.Linear, x.shape[-1], self._units)(x)
            x = self.get(f"norm{index}", NormLayer, x.shape[-1], self._norm)(x)
            x = self._act(x)
        x = x.reshape(features.shape[:-1] + (x.shape[-1],))
        return self.get("out", DistLayer, self._shape, **self._out)(x)


class GRUCell(Module):
    def __init__(self, size, norm=False, act="Tanh", update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = get_act(act)
        self._norm = norm
        self._update_bias = update_bias

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        layer = self.get(
            "gru", nn.Linear, inputs.shape[-1] + state.shape[-1], 3 * self._size
        )
        parts = layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self.get("norm", NormLayer, parts.shape[-1], "layer")(parts)
        reset, cand, update = torch.chunk(parts, 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output


class DistLayer(Module):
    def __init__(self, shape, dist="mse", min_std=0.1, init_std=0.0):
        super().__init__()
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

    def forward(self, inputs):
        out = self.get("out", nn.Linear, inputs.shape[-1], int(np.prod(self._shape)))(
            inputs
        )
        out = torch.reshape(out, inputs.shape[:-1] + self._shape)
        if self._dist in ("normal", "tanh_normal", "trunc_normal"):
            std = self.get("std", nn.Linear, inputs.shape[-1], np.prod(self._shape))(
                inputs
            )
            std = torch.reshape(std, inputs.shape[:-1] + self._shape)
        if self._dist == "mse":
            dist = td.Normal(out, 1.0)
            return ContDist(td.Independent(dist, len(self._shape)))
        if self._dist == "normal":
            dist = td.Normal(out, std)
            return ContDist(td.Independent(dist, len(self._shape)))
        if self._dist == "binary":
            dist = Bernoulli(td.Independent(td.Bernoulli(logits=out), len(self._shape)))
            return dist
        if self._dist == "tanh_normal":
            mean = 5 * torch.tanh(out / 5)
            std = nn.Softplus()(std + self._init_std) + self._min_std
            dist = td.Normal(mean, std)
            dist = td.TransformedDistribution(dist, TanhBijector())
            dist = td.Independent(dist, len(self._shape))
            return SampleDist(dist)
        if self._dist == "trunc_normal":
            std = 2 * torch.sigmoid((std + self._init_std) / 2) + self._min_std
            dist = TruncNormalDist(torch.tanh(out), std, -1, 1)
            return ContDist(td.Independent(dist, 1))
        if self._dist == "onehot":
            return OneHotDist(out)
        raise NotImplementedError(self._dist)


class NormLayer(nn.Module):
    def __init__(self, dim, name):
        super().__init__()
        if name == "none":
            self._layer = None
        elif name == "layer":
            self._layer = nn.LayerNorm(dim)
        else:
            raise NotImplementedError(name)

    def forward(self, features):
        if not self._layer:
            return features
        return self._layer(features)


def get_act(name):
    if name == "none":
        return lambda x: x
    if name == "mish":
        return lambda x: x * torch.tanh(nn.Softplus()(x))
    elif hasattr(torch.nn, name):
        return getattr(torch.nn, name)()
    else:
        raise NotImplementedError(name)
