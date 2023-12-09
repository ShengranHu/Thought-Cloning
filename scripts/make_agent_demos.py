#!/usr/bin/env python3

"""
Generate a set of agent demonstrations.

The agent can either be a trained model or the heuristic expert (bot).

Demonstration generation can take a long time, but it can be parallelized
if you have a cluster at your disposal. Provide a script that launches
make_agent_demos.py at your cluster as --job-script and the number of jobs as --jobs.


"""

import argparse
import gym
import logging
import sys
import subprocess
import os
import time
import numpy as np
import random
import blosc
import torch
import pdb
import babyai.utils as utils

# Parse arguments
# BabyAI-%s-v0

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--env",
    default="BabyAI-BossLevel-v0",
    help="name of the environment to be run (REQUIRED)",
)
parser.add_argument(
    "--model", default="BOT", help="name of the trained model (REQUIRED)"
)
parser.add_argument(
    "--demos",
    default=None,
    help="path to save demonstrations (based on --model and --origin by default)",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=512,
    help="number of episodes to generate demonstrations for",
)
parser.add_argument(
    "--valid-episodes",
    type=int,
    default=0,
    help="number of validation episodes to generate demonstrations for",
)
parser.add_argument("--seed", type=int, default=2023, help="start random seed")
parser.add_argument(
    "--argmax",
    action="store_true",
    default=False,
    help="action with highest probability is selected",
)
parser.add_argument(
    "--log-interval", type=int, default=100, help="interval between progress reports"
)
parser.add_argument(
    "--save-interval",
    type=int,
    default=10000,
    help="interval between demonstrations saving",
)
parser.add_argument(
    "--filter-steps",
    type=int,
    default=0,
    help="filter out demos with number of steps more than filter-steps",
)
parser.add_argument(
    "--on-exception",
    type=str,
    default="warn",
    choices=("warn", "crash"),
    help="How to handle exceptions during demo generation",
)

parser.add_argument(
    "--job-script",
    type=str,
    default=None,
    help="The script that launches make_agent_demos.py at a cluster.",
)
parser.add_argument("--jobs", type=int, default=0, help="job id")
parser.add_argument(
    "--log", action="store_true", default=False, help="Whether save the log"
)
parser.add_argument(
    "--noise-rate", type=float, default=0.0, help="the ratio to inject noise"
)
parser.add_argument("--print", action="store_true", default=False, help="Print demos")

args = parser.parse_args()
# logger = WandbLogger("wandb", )
logger = logging.getLogger(__name__)

color_options = ["red", "green", "blue", "purple", "yellow", "grey"]
item_options = ["key", "ball", "box"]
drop_note_options = [
    "temporary to find the key",
    "temporary to remove blocking object",
    "to complete PutNext mission",
]
pickup_note_options = ["to complete PutNext mission", "to complete pickup mission"]

id2action = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}


def gen_random_subgoal():
    random_sg = ""
    if np.random.random() < (2 / 7):
        # open
        random_sg = (
            "unlock {} door" if np.random.random() < (1 / 10) else "open {} door"
        )
        random_sg.format(random.choice(color_options))
    elif np.random.random() < (2 / 7):
        # drop
        random_sg = "drop {} {}".format(
            random.choice(color_options), random.choice(item_options)
        )
        random_sg += " " + random.choice(drop_note_options)
    elif np.random.random() < (1 / 7):
        # remove
        random_sg = "remove blocking object {} {}".format(
            random.choice(color_options), random.choice(item_options)
        )
    else:
        # pickup
        random_sg = "pickup {} {}".format(
            random.choice(color_options), random.choice(item_options)
        )
        random_sg += " " + random.choice(pickup_note_options)

    return random_sg


# Set seed for all randomness sources


def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    logger.info(
        "Demo length: {:.3f}+-{:.3f}".format(
            np.mean(num_frames_per_episode), np.std(num_frames_per_episode)
        )
    )


def generate_demos(n_episodes, valid, seed, shift=0):
    utils.seed(seed)

    # Generate environment
    env = gym.make("BabyAI-%s-v0" % (args.env))

    action_options = [env.actions.left, env.actions.right, env.actions.forward]

    agent = utils.load_agent(
        env, args.model, args.demos, "agent", args.argmax, args.env
    )
    demos_path = utils.get_demos_path(
        args.demos, args.env, "agent", n_episodes, time_stamp=True, jobs=args.jobs
    )
    demos = []
    logger.info("demos_path: {}".format(demos_path))

    checkpoint_time = time.time()

    just_crashed = False
    while True:
        if len(demos) == n_episodes:
            break

        done = False
        if just_crashed:
            logger.info(
                "reset the environment to find a mission that the bot can solve"
            )
            env.reset()
        else:
            current_seed = seed + (args.jobs * args.episodes) + len(demos)
            env.seed(current_seed)
            if len(demos) and len(demos) % args.log_interval == 0:
                logger.info("seed set to {}".format(current_seed))
        obs = env.reset()
        agent.on_reset()

        actions = []
        mission = obs["mission"]
        images = []
        directions = []
        subgoals = []
        rewards = []

        try:
            action = None
            adding_noise = False
            total_noise_len = 0
            current_len = 0
            noise_subgoal = ""
            while not done:
                agent_plan = agent.act(action_taken=action)
                sg_sentance = agent_plan["sg_sentance"]
                action = agent_plan["action"]
                if isinstance(action, torch.Tensor):
                    action = action.item()

                if not adding_noise and np.random.random() < args.noise_rate:
                    adding_noise = True
                    total_noise_len = np.random.randint(1, 6)
                    current_len = 0
                    noise_subgoal = gen_random_subgoal()

                if adding_noise:
                    current_len += 1
                    if current_len > total_noise_len:
                        adding_noise = False
                        total_noise_len = 0
                        current_len = 0
                    else:
                        action = random.choice(action_options)
                        sg_sentance = noise_subgoal

                new_obs, reward, done, _ = env.step(action)
                agent.analyze_feedback(reward, done)

                actions.append(action)
                subgoals.append(sg_sentance)
                images.append(obs["image"])
                directions.append(obs["direction"])
                rewards.append(reward)
                # pdb.set_trace()

                obs = new_obs
            if reward > 0 and (
                args.filter_steps == 0 or len(images) <= args.filter_steps
            ):
                demos.append(
                    (
                        mission,
                        blosc.pack_array(np.array(images)),
                        directions,
                        actions,
                        subgoals,
                        rewards,
                        current_seed,
                    )
                )
                just_crashed = False
                if args.print:
                    print("===============Episode===================")
                    print("mission: {}".format(mission))
                    last_sg = subgoals[0]
                    last_sg_start = 1
                    last_sg_end = 1
                    actions_cache = [id2action[actions[0]]]

                    T = len(actions)

                    for t in range(1, T):
                        if last_sg == subgoals[t]:
                            last_sg_end = t + 1
                            actions_cache.append(id2action[actions[t]])
                        else:
                            if last_sg_end == last_sg_start:
                                print(
                                    't={}: Thought "{}", Action [{}]'.format(
                                        last_sg_start, last_sg, ",".join(actions_cache)
                                    )
                                )
                            else:
                                print(
                                    't={}-{}: Thought "{}", Action [{}, {}, ... , {}]'.format(
                                        last_sg_start,
                                        last_sg_end,
                                        last_sg,
                                        actions_cache[0],
                                        actions_cache[0],
                                        actions_cache[-1],
                                    )
                                )

                            # empty and update action cache
                            actions_cache = [id2action[actions[t]]]

                            # update last sg and time step
                            last_sg = subgoals[t]
                            last_sg_start = last_sg_end = t + 1

                    if last_sg_end == last_sg_start:
                        print(
                            't={}: Thought "{}", Action [{}]'.format(
                                last_sg_start, last_sg, ",".join(actions_cache)
                            )
                        )
                    else:
                        print(
                            't={}-{}: Thought "{}", Action [{},...,{}]'.format(
                                last_sg_start,
                                last_sg_end,
                                last_sg,
                                actions_cache[0],
                                actions_cache[-1],
                            )
                        )

                    # for step, (act, sg) in enumerate(zip(actions, subgoals)):
                    #     print("Step {}: Thought \"{}\"; Action \"{}\"".format(step, sg, id2action[act]))

            if reward == 0:
                if args.on_exception == "crash":
                    raise Exception(
                        "mission failed, the seed is {}".format(seed + len(demos))
                    )
                just_crashed = True
                logger.info("mission failed")
        except (Exception, AssertionError) as e:
            if args.on_exception == "crash":
                raise
            just_crashed = True
            logger.exception("error while generating demo #{}".format(len(demos)))
            logger.exception(e)
            continue

        if len(demos) and len(demos) % args.log_interval == 0:
            now = time.time()
            demos_per_second = args.log_interval / (now - checkpoint_time)
            to_go = (n_episodes - len(demos)) / demos_per_second
            logger.info(
                "demo #{}, {:.3f} demos per second, {:.3f} seconds to go".format(
                    len(demos) - 1, demos_per_second, to_go
                )
            )
            checkpoint_time = now

        # Save demonstrations

        if (
            args.save_interval > 0
            and len(demos) < n_episodes
            and len(demos) % args.save_interval == 0
        ):
            logger.info("Saving demos...")
            utils.save_demos(demos, demos_path)
            logger.info("{} demos saved".format(len(demos)))
            # print statistics for the last 100 demonstrations
            print_demo_lengths(demos[-100:])

    # Save demonstrations
    logger.info("Saving demos...")
    # pdb.set_trace()
    utils.save_demos(demos, demos_path)
    logger.info("{} demos saved".format(len(demos)))
    print_demo_lengths(demos[-100:])


def generate_demos_cluster():
    demos_per_job = args.episodes // args.jobs
    demos_path = utils.get_demos_path(args.demos, args.env, "agent")
    job_demo_names = [
        os.path.realpath(demos_path + ".shard{}".format(i)) for i in range(args.jobs)
    ]
    for demo_name in job_demo_names:
        job_demos_path = utils.get_demos_path(demo_name)
        if os.path.exists(job_demos_path):
            os.remove(job_demos_path)

    command = [args.job_script]
    command += sys.argv[1:]
    for i in range(args.jobs):
        cmd_i = list(
            map(
                str,
                command
                + ["--seed", args.seed + i * demos_per_job]
                + ["--demos", job_demo_names[i]]
                + ["--episodes", demos_per_job]
                + ["--jobs", 0]
                + ["--valid-episodes", 0],
            )
        )
        logger.info("LAUNCH COMMAND")
        logger.info(cmd_i)
        output = subprocess.check_output(cmd_i)
        logger.info("LAUNCH OUTPUT")
        logger.info(output.decode("utf-8"))

    job_demos = [None] * args.jobs
    while True:
        jobs_done = 0
        for i in range(args.jobs):
            if job_demos[i] is None or len(job_demos[i]) < demos_per_job:
                try:
                    logger.info("Trying to load shard {}".format(i))
                    job_demos[i] = utils.load_demos(
                        utils.get_demos_path(job_demo_names[i])
                    )
                    logger.info(
                        "{} demos ready in shard {}".format(len(job_demos[i]), i)
                    )
                except Exception:
                    logger.exception("Failed to load the shard")
            if job_demos[i] and len(job_demos[i]) == demos_per_job:
                jobs_done += 1
        logger.info("{} out of {} shards done".format(jobs_done, args.jobs))
        if jobs_done == args.jobs:
            break
        logger.info("sleep for 60 seconds")
        time.sleep(60)

    # Training demos
    all_demos = []
    for demos in job_demos:
        all_demos.extend(demos)
    utils.save_demos(all_demos, demos_path)


log_format = "%(asctime)s: %(levelname)s: %(message)s"
logging.basicConfig(level="INFO", format=log_format)
if args.log:
    fh = logging.FileHandler(
        os.path.join(
            utils.storage_dir(),
            "logs",
            "log" + time.strftime("_%m-%d-%H-%M-%S", time.localtime()) + ".txt",
        )
    )
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

logger.info(args)
generate_demos(args.episodes, args.valid_episodes, args.seed)
