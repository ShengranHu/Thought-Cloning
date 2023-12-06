from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn
import copy
import gym
import time
import datetime
import numpy as np
import sys
import itertools
import torch
from babyai.evaluate import TC_batch_evaluate
import babyai.utils as utils
from babyai.rl import DictList
from babyai.TC_models import ThoughCloningModel
from babyai.submodules import maskedNll
import multiprocessing
import os
import json
import logging
import gc
from tqdm import tqdm
import pdb

from babyai.rl.algos import DQNAgent


class AWAC(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )

        self.model = ThoughCloningModel()
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            qa_values = self.critic(observations)
            next_qa_values = self.critic(next_observations)

            # Use the actor to compute a critic backup
            next_qs = rewards + torch.max(next_qa_values, dim=-1)[0] * (
                1 - dones.float()
            )

            # TODO(student): Compute the TD target
            target_values = rewards + self.discount * next_qs

        # TODO(student): Compute Q(s, a) and loss similar to DQN
        q_values = torch.gather(
            self.critic(observations), -1, actions.unsqueeze(1)
        ).squeeze()
        assert q_values.shape == target_values.shape

        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        if action_dist is not None:
            action_dist = action_dist.sample()
        qa_values = self.critic(observations)
        q_values = torch.gather(qa_values, -1, actions.unsqueeze(1)).squeeze()
        values = torch.max(qa_values, dim=-1)[0]
        advantages = q_values - values
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        adv = self.compute_advantage(observations, actions)
        loss = (
            -self.actor(observations).log_prob(actions)
            * torch.exp(adv / self.temperature)
            * self.temperature
        )
        loss = loss.mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics = super().update(
            observations, actions, rewards, next_observations, dones, step
        )

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
