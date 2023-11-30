from typing import Optional, Sequence
import numpy as np
import torch
import pdb

import babyai.utils.ptu as ptu
from torch import nn


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass of the critic network
        return self.network(obs)

    def update(
        self, obs: np.ndarray, rewards: np.ndarray, q_values: np.ndarray
    ) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)
        rewards = ptu.from_numpy(rewards)

        # TODO: update the critic using the observations and q_values
        self.optimizer.zero_grad()
        val_est = self.forward(obs).squeeze()
        # target = torch.concat([q_values[:-1], torch.tensor([0])]) + rewards
        target = q_values + rewards
        loss = nn.MSELoss()(target, val_est)
        loss.backward()
        self.optimizer.step()

        return {
            "Baseline Loss": ptu.to_numpy(loss),
        }


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        obs = torch.tensor(obs, device=ptu.device)
        action = ptu.to_numpy(self.forward(obs))

        return action

    def forward(self, obs: torch.FloatTensor) -> torch.distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            dist = torch.distributions.Categorical(nn.Softmax()(self.logits_net(obs)))
        else:
            dist = torch.distributions.Normal(
                self.mean_net(obs), torch.exp(self.logstd)
            )
        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        dists = self.forward(obs).log_prob(actions)
        if len(dists.shape) == 1:
            dists = dists.reshape(-1, 1)
        dists = torch.mean(dists, -1)
        # TODO: implement the policy gradient actor update.
        loss = -torch.mean(dists * advantages)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {
            "Actor Loss": ptu.to_numpy(loss),
        }


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        q_values = np.concatenate(q_values, axis=0)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(
            obs,
            actions,
            advantages,
        )

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            for i in range(self.baseline_gradient_steps):
                critic_info: dict = self.critic.update(obs, rewards, q_values)
                info.update(critic_info)

        return info

    def forward(self, obs):
        return self.actor(obs)

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values = [self._discounted_return(r) for r in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = [self._discounted_reward_to_go(r) for r in rewards]

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            advantages = q_values
        else:
            # TODO: run the critic and use it as a baseline
            values = ptu.to_numpy(self.critic.forward(ptu.from_numpy(obs)).squeeze())
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    advantages[i] = rewards[i] + self.gamma * (
                        (1 - self.gae_lambda) * values[i + 1]
                        + self.gae_lambda * advantages[i + 1]
                    )

                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            mean = np.average(advantages)
            std = np.std(advantages - mean)
            if np.abs(std - 0) <= 0.0001:
                std = 1
            advantages = (advantages - mean) / std

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        dr = 0
        for i, r in enumerate(rewards[::-1]):
            dr += r * np.power(self.gamma, i)
        return np.repeat(dr, len(rewards))

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        out = []
        for i, r in enumerate(rewards[::-1]):
            if i == 0:
                out.append(r)
            else:
                out.append(r + out[-1] * self.gamma)
        return out[::-1]
