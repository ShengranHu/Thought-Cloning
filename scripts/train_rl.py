#!/usr/bin/env python3

"""
Script to train agent through imitation learning using demonstrations.
"""

import os
import csv
import copy
import gym
import time
import datetime
import numpy as np
import sys
import logging
import torch
from babyai.arguments import ArgumentParser
import babyai.utils as utils
from babyai.thought_cloning import OfflineLearning, OfflineLanguageLearning
import pdb

# Parse arguments
parser = ArgumentParser()
parser.add_argument(
    "--device",
    type=int,
    default=None,
    help="which GPU",
)
parser.add_argument(
    "--demos",
    default=None,
    help="demos filename (REQUIRED or demos-origin or multi-demos required)",
)
parser.add_argument(
    "--demos-origin",
    required=False,
    help="origin of the demonstrations: human | agent (REQUIRED or demos or multi-demos required)",
)
parser.add_argument(
    "--valid",
    help="file path for validation",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=0,
    help="number of episodes of demonstrations to use"
    "(default: 0, meaning all demos)",
)
parser.add_argument(
    "--multi-env",
    nargs="*",
    default=None,
    help="name of the environments used for validation/model loading",
)
parser.add_argument(
    "--multi-demos",
    nargs="*",
    default=None,
    help="demos filenames for envs to train on (REQUIRED when multi-env is specified)",
)
parser.add_argument(
    "--multi-episodes",
    type=int,
    nargs="*",
    default=None,
    help="number of episodes of demos to use from each file (REQUIRED when multi-env is specified)",
)
parser.add_argument(
    "--save-interval",
    type=int,
    default=1,
    help="number of epochs between two saves (default: 1, 0 means no saving)",
)
parser.add_argument(
    "--sg-coef", type=float, default=2.0, help="sg loss term coefficient (default: 2)"
)
parser.add_argument(
    "--warm-start", action="store_true", default=False, help="have 5 warmup epochs"
)
parser.add_argument("--expr-group-name", default="", help="expr group name")
parser.add_argument(
    "--target-update-period",
    default=16,
    type=int,
    help="how many steps before updating target",
)
parser.add_argument("--language", action="store_true")
parser.add_argument("--lower-only", action="store_true")
parser.add_argument("--wandb-id", help="log name for WandB")


def main(args):
    # pdb.set_trace()
    if args.device is not None:
        torch.cuda.set_device(args.device)
    # Verify the arguments when we train on multiple environments
    # No need to check for the length of len(args.multi_env) in case, for some reason, we need to validate on other envs
    if args.multi_env is not None:
        assert len(args.multi_demos) == len(args.multi_episodes)

    utils.group_name = args.expr_group_name

    if args.language:
        args.model = args.model or OfflineLanguageLearning.default_model_name(args)
    else:
        args.model = args.model or OfflineLearning.default_model_name(args)
    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    rl_learn = OfflineLanguageLearning(args) if args.language else OfflineLearning(args)

    # Define logger and Tensorboard writer
    header = [
        "update",
        "frames",
        "FPS",
        "duration",
        "entropy",
        "policy_loss",
        "sg_loss",
        "train_accuracy",
    ] + ["validation_accuracy"]
    if args.multi_env is None:
        header.extend(["validation_return", "validation_success_rate"])
    else:
        header.extend(["validation_return_{}".format(env) for env in args.multi_env])
        header.extend(
            ["validation_success_rate_{}".format(env) for env in args.multi_env]
        )
    writer = None
    if args.tb:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(utils.get_log_dir(args.model))

    # Define csv writer
    csv_writer = None
    csv_path = os.path.join(utils.get_log_dir(args.model), "log.csv")
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer that one write to the log
    csv_writer = csv.writer(open(csv_path, "a", 1))
    if first_created:
        csv_writer.writerow(header)
    logger.info("prepared csv writer")

    # Get the status path
    status_path = os.path.join(utils.get_log_dir(args.model), "status.json")
    logger.info("read status")

    # Log command, availability of CUDA, and model
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))
    logger.info(rl_learn.acmodel)

    rl_learn.train(rl_learn.train_demos, writer, csv_writer, status_path, header)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
