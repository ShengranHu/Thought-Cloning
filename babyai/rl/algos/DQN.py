from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import babyai.rl.utils.pytorch_util as ptu
import pdb


class DQN(nn.Module):
    def __init__(
        self,
        discount: float,
        target_update_period: int,
        critic_optimizer,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q
        self.critic_optimizer = critic_optimizer

        self.critic_loss = nn.MSELoss()

    def compute_critic_loss(
        self,
        qa_values: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_qa_values: torch.Tensor,
        done: torch.Tensor,
        double_next_qa_values: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict, dict]:
        """
        Compute the loss for the DQN critic.

        Returns:
         - loss: torch.Tensor, the MSE loss for the critic
         - metrics: dict, a dictionary of metrics to log
         - variables: dict, a dictionary of variables that can be used in subsequent calculations
        """
        (batch_size,) = reward.shape
        # TODO(student): paste in your code from HW3, and make sure the return values exist
        with torch.no_grad():
            # TODO(student): compute target values
            """
            next_qa_values = ...

            if self.use_double_q:
                raise NotImplementedError
            else:
                next_action = ...

            next_q_values = ...
            target_values = ...
            """
            if double_next_qa_values is not None:
                # Double-Q
                doubleq_next_action = double_next_qa_values.argmax(dim=-1)
                next_q_values = torch.gather(
                    next_qa_values, 1, doubleq_next_action.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q_values, _ = next_qa_values.max(dim=-1)

            assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values: torch.Tensor = reward + self.discount * next_q_values * (
                1 - done.float()
            )
            assert target_values.shape == (batch_size,), target_values.shape

        # Select Q-values for the actions that were actually taken
        q_values = torch.gather(qa_values, 1, action.unsqueeze(1)).squeeze(1)
        assert q_values.shape == (batch_size,), q_values.shape

        # Compute loss
        loss: torch.Tensor = self.critic_loss(q_values, target_values)
        # ENDTODO

        # self.critic_optimizer.zero_grad()
        # loss.backward()
        # grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
        #     self.critic.parameters(), self.clip_grad_norm or float("inf")
        # )
        # self.critic_optimizer.step()

        # self.lr_scheduler.step()
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

    def update_critic(
        self,
        qa_values: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_qa_values: torch.Tensor,
        done: torch.Tensor,
        double_next_qa_values=None,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        loss, metrics, _ = self.compute_critic_loss(
            qa_values,
            action,
            reward,
            next_qa_values,
            done,
            double_next_qa_values,
        )

        # self.critic_optimizer.zero_grad()
        # loss.backward()
        # grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
        #     self.critic.parameters(), self.clip_grad_norm or float("inf")
        # )
        # metrics["grad_norm"] = grad_norm.item()
        # self.critic_optimizer.step()

        return loss, metrics

    def update(
        self,
        qa_values: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_qa_values: torch.Tensor,
        done: torch.Tensor,
        double_next_qa_values=None,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): paste in your code from HW3

        # TODO(student): update the critic, and the target if needed
        loss, critic_stats = self.update_critic(
            qa_values,
            action,
            reward,
            next_qa_values,
            done,
            double_next_qa_values,
        )
        # ENDTODO
        return loss, critic_stats
