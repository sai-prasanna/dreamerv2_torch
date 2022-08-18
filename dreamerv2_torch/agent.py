from typing import Callable, Dict
import torch
import torch.nn as nn
import torch.distributions as td
from torch import Tensor

import common
import expl

class Agent(nn.Module):
    def __init__(self, config, obs_space, act_space, step):
        super().__init__()
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.tfstep = nn.Parameter(torch.ones(()) * int(self.step), requires_grad=False)
        self.wm = WorldModel(config, obs_space, self.act_space, self.tfstep)
        self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)
        if config.expl_behavior == "greedy":
            self._expl_behavior = self._task_behavior
        else:
            self._expl_behavior = getattr(expl, config.expl_behavior)(
                self.config,
                self.act_space,
                self.wm,
                self.tfstep,
                lambda seq: self.wm.heads["reward"](seq["feat"]).mode(),
            )

    def policy(self, obs, state=None, mode="train"):
        obs = self.wm.preprocess(obs)
        self.tfstep.copy_(torch.tensor([int(self.step)])[0])
        if state is None:
            latent = self.wm.rssm.initial(len(obs["reward"]))
            action = torch.zeros((len(obs["reward"]),) + self.act_space.shape).to(obs["reward"].device)
            state = latent, action
        latent, action = state
        embed = self.wm.encoder(obs)
        sample = (mode == "train") or not self.config.eval_state_mean
        latent, _ = self.wm.rssm.obs_step(
            latent, action, embed, obs["is_first"], sample
        )
        feat = self.wm.rssm.get_feature(latent)
        if mode == "eval":
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
            noise = self.config.eval_noise
        elif mode == "explore":
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise
        elif mode == "train":
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise
        action = action.detach()
        action = common.action_noise(action, noise, self.act_space)
        latent = {k: v.detach() for k, v in latent.items()}
        outputs = {"action": action}
        state = (latent, action)
        return outputs, state

    def train(self, data, state=None):
        metrics = {}
        state, outputs, mets = self.wm.train(data, state)
        metrics.update(mets)
        start = outputs["post"]
        reward = lambda seq: self.wm.heads["reward"](seq["feat"]).mode()
        metrics.update(
            self._task_behavior.train(self.wm, start, data["is_terminal"], reward)
        )
        if self.config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, outputs, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        return state, metrics

    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        for key in self.wm.heads["decoder"].cnn_keys:
            name = key.replace("/", "_")
            report[f"openl_{name}"] = self.wm.video_pred(data, key)
        return report


class WorldModel(nn.Module):
    def __init__(self, config, obs_space, action_space, tfstep):
        super().__init__()
        self._use_amp = True if config.precision == 16 else False
        self.action_size = action_space.shape[0]
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.tfstep = tfstep
        self.encoder = common.Encoder(shapes, **config.encoder)

        self.rssm = common.EnsembleRSSM(
            self.action_size, self.encoder.embed_size, **config.rssm
        )
        self.heads = {}
        self.heads["decoder"] = common.Decoder(
            shapes,
            self.rssm.state_dim,
            self.encoder.cnn_output_shapes,
            config.encoder.cnn_kernels,
            config.encoder.mlp_layers,
            config.encoder.act,
            config.encoder.norm,
            config.encoder.cnn_keys,
            config.encoder.mlp_keys,
        )
        self.heads["reward"] = common.MLPDistribution(
            self.rssm.state_dim, 1, **config.reward_head
        )

        if config.pred_discount:
            self.heads["discount"] = common.MLPDistribution(
                self.rssm.state_dim, 1, **config.discount_head
            )
        for name in config.grad_heads:
            assert name in self.heads, name
        self.heads = nn.ModuleDict(self.heads)
        self.model_opt = common.Optimizer(
            "model", self.parameters(), **config.model_opt, use_amp=self._use_amp 
        )

    def train(self, data, state=None):
        with common.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                model_loss, state, outputs, metrics = self.loss(data, state)
            metrics.update(self.model_opt(model_loss))
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data["action"], data["is_first"], state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {"kl": kl_loss}
        feat = self.rssm.get_feature(post)
        for name, head in self.heads.items():
            grad_head = name in self.config.grad_heads
            inp = feat if grad_head else feat.detach()
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                like = dist.log_prob(data[key]).to(torch.float32)
                likes[key] = like
                losses[key] = -like.mean()
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        detach_dict = lambda x: {k: v.detach() for k, v in x.items()}
        outs = dict(
            embed=embed, feat=feat, post=detach_dict(post), prior=prior, likes=likes, kl=kl_value
        )
        metrics = {f"{name}_loss": value.detach().cpu() for name, value in losses.items()}
        metrics["model_kl"] = kl_value.mean().detach().cpu()
        metrics["prior_ent"] = self.rssm._get_dist(prior).entropy().mean().detach().cpu()
        metrics["post_ent"] = self.rssm._get_dist(post).entropy().mean().detach().cpu()
        last_state = {k: v[:, -1].detach() for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    def imagine(
        self,
        policy: Callable[[Tensor], "td.Distribution"],
        start_state: Dict[str, Tensor],
        is_terminal: Tensor,
        horizon: int,
    ) -> Tensor:
        """Given a batch of states, rolls out more state of length horizon."""
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start_state.items()}
        start["feat"] = self.rssm.get_feature(start)
        start["action"] = torch.zeros_like(policy(start["feat"]).mode())
        seq = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            action = policy(seq["feat"][-1].detach()).rsample()
            state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feature = self.rssm.get_feature(state)
            for key, value in {**state, "action": action, "feat": feature}.items():
                seq[key].append(value)

        seq = {k: torch.stack(v, 0) for k, v in seq.items()}

        if "discount" in self.heads:
            disc = self.heads["discount"](seq["feat"]).mean()
            if is_terminal is not None:
                true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
                true_first *= self.config.discount
                disc = torch.concat([true_first[None], disc[1:]], 0)
        else:
            disc = self.config.discount * torch.ones(
                seq["feat"].shape[:-1], device=seq["feat"].device
            )
        seq["discount"] = disc.unsqueeze(-1)
        seq["weight"] = torch.cumprod(
            torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0
        ).unsqueeze(-1).detach()
        return seq

    def preprocess(self, obs):
        #dtype = torch.float16 if self.config.precision == 16 else torch.float32
        obs = obs.copy()
        obs = {k: torch.Tensor(v).to(self.config.device) for k, v in obs.items()}
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype == torch.int32:
                value = value#.to(dtype)
            if value.dtype == torch.uint8:
                value = value / 255.0 - 0.5 # value.to(dtype) / 255.0 - 0.5
            obs[key] = value
        obs["reward"] = {
            "identity": lambda x: x,
            "sign": torch.sign,
            "tanh": torch.tanh,
        }[self.config.clip_rewards](obs["reward"]).unsqueeze(-1)
        obs["discount"] = 1.0 - obs["is_terminal"].float().unsqueeze(-1)#.to(dtype)
        obs["discount"] *= self.config.discount
        return obs

    def video_pred(self, data, key):
        decoder = self.heads["decoder"]
        truth = data[key][:6] + 0.5
        embed = self.encoder(data)
        states, _ = self.rssm.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = decoder(self.rssm.get_feature(states))[key].mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data["action"][:6, 5:], init)
        openl = decoder(self.rssm.get_feature(prior))[key].mode()
        model = torch.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = torch.concat([truth, model, error], 2)
        B, T, H, W, C = video.shape
        return video.permute(1, 2, 0, 3, 4).reshape(T, H, B * W, C).cpu()


class ActorCritic(nn.Module):
    def __init__(self, config, act_space, tfstep):
        super().__init__()
        self.config = config
        self._use_amp = True if config.precision == 16 else False
        self.act_space = act_space
        self.tfstep = tfstep
        discrete = hasattr(act_space, "n")
        if self.config.actor.dist == "auto":
            self.config = self.config.update(
                {"actor.dist": "onehot" if discrete else "trunc_normal"}
            )
        if self.config.actor_grad == "auto":
            self.config = self.config.update(
                {"actor_grad": "reinforce" if discrete else "dynamics"}
            )
        if config.rssm.discrete:
            feat_size = config.rssm.stoch * config.rssm.discrete + config.rssm.deter
        else:
            feat_size = config.rssm.stoch + config.rssm.deter
        self.actor = common.MLPDistribution(
            feat_size, act_space.shape[0], **self.config.actor
        )
        self.critic = common.MLPDistribution(feat_size, 1, **self.config.critic)
        if self.config.slow_target:
            self._target_critic = common.MLPDistribution(
                feat_size, 1, **self.config.critic
            )
            self._updates = nn.Parameter(torch.zeros(()), requires_grad=False)
        else:
            self._target_critic = self.critic
        self.actor_opt = common.Optimizer(
            "actor", self.actor.parameters(), use_amp=self._use_amp, **self.config.actor_opt
        )
        self.critic_opt = common.Optimizer(
            "critic", self.critic.parameters(), use_amp=self._use_amp, **self.config.critic_opt
        )
        self.rewnorm = common.StreamNorm(**self.config.reward_norm)

    def train(self, world_model: WorldModel, start, is_terminal, reward_fn):
        metrics = {}
        hor = self.config.imag_horizon
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.
        with common.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                seq = world_model.imagine(self.actor, start, is_terminal, hor)
                reward = reward_fn(seq)
                seq["reward"], mets1 = self.rewnorm(reward)
                mets1 = {f"reward_{k}": v for k, v in mets1.items()}
                target, mets2 = self.target(seq)
                actor_loss, mets3 = self.actor_loss(seq, target)
        with common.RequiresGrad(self.critic):
            with torch.cuda.amp.autocast(self._use_amp):
                critic_loss, mets4 = self.critic_loss(seq, target)
        with common.RequiresGrad(self):
            metrics.update(self.actor_opt(actor_loss))
            metrics.update(self.critic_opt(critic_loss))
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()  # Variables exist after first forward pass.
        return metrics

    def actor_loss(self, seq, target):
        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3
        # Targets:     t0   [t1]  [t2]
        # Baselines:  [v0]  [v1]   v2    v3
        # Entropies:        [e1]  [e2]
        # Weights:    [ 1]  [w1]   w2    w3
        # Loss:              l1    l2
        metrics = {}
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.
        policy = self.actor(seq["feat"][:-2].detach())
        if self.config.actor_grad == "dynamics":
            objective = target[1:]
        elif self.config.actor_grad == "reinforce":
            baseline = self._target_critic(seq["feat"][:-2]).mode()
            advantage = (target[1:] - baseline).detach()
            action = seq["action"][1:-1].detach()
            objective = policy.log_prob(action) * advantage
        elif self.config.actor_grad == "both":
            baseline = self._target_critic(seq["feat"][:-2]).mode()
            advantage = (target[1:] - baseline).detach()
            objective = policy.log_prob(seq["action"][1:-1]) * advantage
            mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
            objective = mix * target[1:] + (1 - mix) * objective
            metrics["actor_grad_mix"] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy().unsqueeze(-1)
        ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
        objective = objective + ent_scale * ent
        weight = seq["weight"].detach()
        actor_loss = -(weight[:-2] * objective).mean()
        metrics["actor_ent"] = ent.mean().detach().cpu()
        metrics["actor_ent_scale"] = ent_scale
        return actor_loss, metrics

    def critic_loss(self, seq, target):
        # States:     [z0]  [z1]  [z2]   z3
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]   v3
        # Weights:    [ 1]  [w1]  [w2]   w3
        # Targets:    [t0]  [t1]  [t2]
        # Loss:        l0    l1    l2
        dist = self.critic(seq["feat"][:-1].detach())
        target = target.detach()
        weight = seq["weight"].detach()
        critic_loss = -(dist.log_prob(target).unsqueeze(-1) * weight[:-1]).mean()
        metrics = {"critic": dist.mode().mean().detach().cpu()}
        return critic_loss, metrics

    def target(self, seq):
        # States:     [z0]  [z1]  [z2]  [z3]
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]  [v3]
        # Discount:   [d0]  [d1]  [d2]   d3
        # Targets:     t0    t1    t2
        reward = seq["reward"].to(torch.float32)
        disc = seq["discount"].to(torch.float32)
        value = self._target_critic(seq["feat"]).mode()
        # Skipping last time step because it is used for bootstrapping.
        target = common.lambda_return(
            reward[:-1],
            value[:-1],
            disc[:-1],
            bootstrap=value[-1],
            lambda_=self.config.discount_lambda,
        )
        metrics = {}
        metrics["critic_slow"] = value.mean().detach().cpu()
        metrics["critic_target"] = target.mean().detach().cpu()
        return target, metrics

    def update_slow_target(self):
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = (
                    1.0
                    if self._updates == 0
                    else float(self.config.slow_target_fraction)
                )
                for s, d in zip(
                    self.critic.parameters(), self._target_critic.parameters()
                ):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
