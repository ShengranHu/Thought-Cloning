import argparse
import gym

import babyai.utils as utils
from babyai.evaluate import ManyEnvs

import numpy as np
import os
import pickle
import random
import logging
import csv
from babyai.levels.verifier import *
import pdb

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env", required=True, help="name of the environment to be run (REQUIRED)"
)
parser.add_argument(
    "--model", default=None, required=True, help="name of the trained model (REQUIRED)"
)
parser.add_argument(
    "--testing-levels-path", required=True, help="path to testing levels (REQUIRED)"
)
parser.add_argument("--expr-group-name", default="", help="expr group name")


def TC_batch_evaluate(agent, env_name, seeds, return_obss_actions=False):
    num_envs = len(seeds)
    # pdb.set_trace()
    # print("using default evalutaion: 512 many envs + no decode subgoal + no.cpu()")
    envs = []
    for i in range(num_envs):
        env = gym.make("BabyAI-%s-v0" % (env_name))
        envs.append(env)

    env = ManyEnvs(envs)

    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "observations_per_episode": [],
        "actions_per_episode": [],
        "seed_per_episode": [],
        "subgoals_per_episode": [],
        "visualization_per_episode": [],
    }

    env.seed(seeds)

    many_obs = env.reset()

    cur_num_frames = 0
    num_frames = np.zeros((num_envs,), dtype="int64")
    returns = np.zeros((num_envs,))
    already_done = np.zeros((num_envs,), dtype="bool")

    if return_obss_actions:
        obss = [[] for _ in range(num_envs)]
        actions = [[] for _ in range(num_envs)]
        subgoals = [[] for _ in range(num_envs)]

    while (num_frames == 0).any():
        results = agent.act_batch(many_obs)
        action = results["action"]

        if return_obss_actions:
            subgoal = results["subgoal"]
            # frames = env.render('rgb_array', tile_size=32)
            for i in range(num_envs):
                if not already_done[i]:
                    obss[i].append(many_obs[i])
                    actions[i].append(action[i].item())
                    subgoals[i].append(subgoal[i])

        many_obs, reward, done, _ = env.step(action)

        agent.analyze_feedback(reward, done)
        done = np.array(done)
        just_done = done & (~already_done)
        returns += reward * just_done
        cur_num_frames += 1
        num_frames[just_done] = cur_num_frames
        already_done[done] = True

    logs["num_frames_per_episode"].extend(list(num_frames))
    logs["return_per_episode"].extend(list(returns))
    logs["seed_per_episode"].extend(list(seeds))
    if return_obss_actions:
        logs["observations_per_episode"].extend(obss)
        logs["actions_per_episode"].extend(actions)
        logs["subgoals_per_episode"].extend(subgoals)

    return logs


def main(args, seeds):
    env = gym.make("BabyAI-%s-v0" % (args.env))
    agent = utils.load_agent(env, args.model, argmax=False, TC=True)

    # Evaluate
    logs = TC_batch_evaluate(agent, args.env, seeds, return_obss_actions=True)
    return logs


if __name__ == "__main__":
    # fix random seed
    random.seed(0)

    args = parser.parse_args()
    utils.group_name = args.expr_group_name

    utils.configure_logging("Zero-shot_evaluation-{}".format(args.model))
    logger = logging.getLogger(__name__)
    logger.info(args)

    with open(args.testing_levels_path, "rb") as f:
        testing_levels = pickle.load(f)

    csv_path = os.path.join(
        utils.storage_dir(),
        utils.expr_group_name(),
        "zeroshot_{}.csv".format(args.model),
    )
    first_created = not os.path.exists(csv_path)
    csv_writer = csv.writer(open(csv_path, "w", 1))
    if first_created:
        csv_writer.writerow(["diff_type", "diff", "acc"])
    logger.info("prepared csv writer")

    for diff_type in ["demo", "mission"]:
        for difficulty in testing_levels[diff_type].keys():
            logger.info("{}: {}".format(diff_type, difficulty))
            seeds = list(testing_levels[diff_type][difficulty].keys())
            logs = main(args, seeds)
            success_per_episode = utils.synthesize(
                [1 if r > 0 else 0 for r in logs["return_per_episode"]]
            )
            csv_writer.writerow([diff_type, difficulty, success_per_episode["mean"]])
            logger.info("diff {} | {}".format(difficulty, success_per_episode["mean"]))
