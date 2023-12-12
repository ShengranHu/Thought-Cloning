from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import babyai.rl.utils.pytorch_util as ptu
from babyai.rl.algos.DQN import DQN


class CQL(DQN):
    def __init__(
        self,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature

    def compute_critic_loss(
        self,
        qa_values: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_qa_values: torch.Tensor,
        done: bool,
        double_next_qa_values: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict, dict]:
        loss, metrics, variables = super().compute_critic_loss(
            qa_values,
            action,
            reward,
            next_qa_values,
            done,
            double_next_qa_values,
        )

        # TODO(student): modify the loss to implement CQL
        qa_values = variables['qa_values'] 
        q_values = variables['q_values']    
        # print('q', q_values)
        # print('qa', qa_values)
        log_sum_exp_q_values = torch.logsumexp(qa_values/self.cql_temperature, dim=1)
        # print('sum_exp', log_sum_exp_q_values)


        cql_reg = log_sum_exp_q_values*self.cql_temperature - q_values
        # print('cql', cql_reg)
        cql_reg = self.cql_alpha * cql_reg.mean()
        # print('cql alpha', cql_reg)

        loss += cql_reg

        return loss, metrics, variables
    
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
        critic_loss, metrics = self.update_critic(
            qa_values,
            action,
            reward,
            next_qa_values,
            done,
            double_next_qa_values,
        )

        metrics["critic_loss"] = critic_loss

        return metrics