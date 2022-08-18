import collections
import contextlib
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td

from dreamerv2_torch.common import dists


class RandomAgent:
    def __init__(self, act_space, logprob=False):
        self.act_space = act_space["action"]
        self.logprob = logprob
        if hasattr(self.act_space, "n"):
            self._dist = dists.OneHotDist(torch.zeros(self.act_space.n))
        else:
            dist = td.Uniform(
                torch.tensor(self.act_space.low), torch.tensor(self.act_space.high)
            )
            self._dist = td.Independent(dist, 1)

    def __call__(self, obs, state=None, mode=None):
        action = self._dist.sample((len(obs["is_first"]),))
        output = {"action": action}
        if self.logprob:
            output["logprob"] = self._dist.log_prob(action)
        return output, None


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        step = step.to(torch.float32)
        match = re.match(r"linear\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clip(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r"warmup\((.+),(.+)\)", string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.clip(step / warmup, 0, 1)
            return scale * value
        match = re.match(r"exp\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r"horizon\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clip(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)


def lambda_return(reward, value, weights, bootstrap, lambda_):
    def agg_fn(x, y):
        return y[0] + y[1] * lambda_ * x

    next_values = torch.cat([value[1:], bootstrap[None]], dim=0)
    inputs = reward + weights * next_values * (1 - lambda_)

    last = bootstrap
    returns = []
    for i in reversed(range(len(inputs))):
        last = agg_fn(last, [inputs[i], weights[i]])
        returns.append(last)

    returns = list(reversed(returns))
    returns = torch.stack(returns, dim=0)
    return returns


def action_noise(action, amount, act_space):
    if amount == 0:
        return action
    amount = amount.to(action.dtype)
    if hasattr(act_space, "n"):
        probs = amount / action.shape[-1] + (1 - amount) * action
        return dists.OneHotDist(probs=probs).sample()
    else:
        return torch.clip(td.Normal(action, amount).sample(), -1, 1)


class StreamNorm(nn.Module):
    def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8):
        # Momentum of 0 normalizes only based on the current batch.
        # Momentum of 1 disables normalization.
        super().__init__()
        self._shape = tuple(shape)
        self._momentum = momentum
        self._scale = scale
        self._eps = eps
        self.mag = nn.Parameter(
            torch.ones(shape, dtype=torch.float64), requires_grad=False
        )

    def forward(self, inputs):
        metrics = {}
        self.update(inputs)
        metrics["mean"] = inputs.mean().detach().cpu()
        metrics["std"] = inputs.std().detach().cpu()
        outputs = self.transform(inputs)
        metrics["normed_mean"] = outputs.mean().detach().cpu()
        metrics["normed_std"] = outputs.std().detach().cpu()
        return outputs, metrics

    def reset(self):
        self.mag.data = torch.ones_like(self.mag)

    def update(self, inputs):
        batch = inputs.reshape((-1,) + self._shape)
        mag = torch.abs(batch).mean(0).type(torch.float64)
        self.mag.data = self._momentum * self.mag + (1 - self._momentum) * mag

    def transform(self, inputs):
        values = inputs.reshape((-1,) + self._shape)
        values /= self.mag.data.type(inputs.dtype)[None] + self._eps
        values *= self._scale
        return values.reshape(inputs.shape)


class Timer:
    def __init__(self):
        self._indurs = collections.defaultdict(list)
        self._outdurs = collections.defaultdict(list)
        self._start_times = {}
        self._end_times = {}

    @contextlib.contextmanager
    def section(self, name):
        self.start(name)
        yield
        self.end(name)

    def wrap(self, function, name):
        def wrapped(*args, **kwargs):
            with self.section(name):
                return function(*args, **kwargs)

        return wrapped

    def start(self, name):
        now = time.time()
        self._start_times[name] = now
        if name in self._end_times:
            last = self._end_times[name]
            self._outdurs[name].append(now - last)

    def end(self, name):
        now = time.time()
        self._end_times[name] = now
        self._indurs[name].append(now - self._start_times[name])

    def result(self):
        metrics = {}
        for key in self._indurs:
            indurs = self._indurs[key]
            outdurs = self._outdurs[key]
            metrics[f"timer_count_{key}"] = len(indurs)
            metrics[f"timer_inside_{key}"] = np.sum(indurs)
            metrics[f"timer_outside_{key}"] = np.sum(outdurs)
            indurs.clear()
            outdurs.clear()
        return metrics


class CarryOverState:
    def __init__(self, fn):
        self._fn = fn
        self._state = None

    def __call__(self, *args):
        self._state, out = self._fn(*args, self._state)
        return out

def to_np(x: torch.Tensor):
    return x.detach().cpu()