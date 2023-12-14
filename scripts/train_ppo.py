#!/usr/bin/env python3

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
from babyai.TC_models import ThoughCloningModel
from babyai.rl.algos.ppo_tc import PPOAlgo
from babyai.arguments import ArgumentParser
import babyai.utils as utils
import wandb
from babyai.evaluate import TC_batch_evaluate

parser = ArgumentParser()
parser.add_argument(
    "--device",
    type=int,
    default=None,
    help="which GPU",
)
parser.add_argument(
    "--env_num",
    type=int,
    default=1,
    help="Number of environments you want to train",
)
parser.add_argument(
    "--ppo_updates",
    type=int,
    default=10,
    help="Number of PPO Updates you want to run",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="default_model",
    help="Number of environments you want to train",
)
parser.add_argument("--lower-only", action="store_true")

parser.add_argument("--hybrid-warmstart",action="store_true")

def main(args):
    wandb.login()
    if not os.path.exists("models"):
        os.makedirs("models")
    if args.device is not None:
        torch.cuda.set_device(args.device)
    env_list = ["BabyAI-%s-v0" % (args.env)] * args.env_num
    env = [gym.make(item) for item in env_list]
    observation_space = env[0].observation_space
    action_space = env[0].action_space
    if(args.model is None):
        utils.configure_logging(args.model_name)
        logger = logging.getLogger(__name__)
        obss_preprocessor = utils.TCObssPreprocessor(args.model_name, observation_space)
        logger.info("Creating new model")
        acmodel = ThoughCloningModel(
                    obss_preprocessor.obs_space,
                    action_space,
                    args.image_dim,
                    args.memory_dim,
                    args.instr_dim,
                )
        
        model_name = args.model_name
    else:
        utils.configure_logging(args.model_name)
        logger = logging.getLogger(__name__)
        acmodel = torch.load(args.model,map_location="cuda:"+str(args.device))
        load_vocab_from = args.model[:args.model.rfind('/')]
        obss_preprocessor = utils.TCObssPreprocessor(args.model, observation_space,load_vocab_from,ppo_warmstart=True)
        model_name = args.model_name
    acmodel.to(args.device)
    ppo_algo = PPOAlgo(env,acmodel,model_name,lower_only = args.lower_only,hybrid_warmstart=args.hybrid_warmstart)
    run = wandb.init(project="thought_cloning_ppo",config={"seed":args.seed})
    return_per_episodes = []
    reshaped_return_per_episodes = []
    num_frames_per_episodes = []
    num_frameses = []
    episodes_dones = []
    entropys = []
    values = []
    policy_losses = []
    value_losses = []
    grad_norms = []
    losses = []
    min_loss = 100000
    avg_return = -0.001
    best_acmodel = None
    for i in range(args.ppo_updates):
        log = ppo_algo.update_parameters()
        # return_per_episodes.append(log["return_per_episode"])
        # reshaped_return_per_episodes.append(log["reshaped_return_per_episode"])
        # num_frames_per_episodes.append(log["num_frames_per_episode"])
        # num_frameses.append(log["num_frames"])
        # episodes_dones.append(log["episodes_done"])
        # entropys.append(log["entropy"])
        # values.append(log["value"])
        # policy_losses.append(log["policy_loss"])
        # value_losses.append(log["value_loss"])
        # grad_norms.append(log["grad_norm"])
        # losses.append(log["loss"])
        if(sum(log["return_per_episode"]) / len(log["return_per_episode"]) > avg_return):
            avg_return = sum(log["return_per_episode"]) / len(log["return_per_episode"])
            best_acmodel = copy.deepcopy(ppo_algo.acmodel)
            torch.save(ppo_algo.acmodel.state_dict(), "models/" + args.model_name + ".pt")
            wandb.save("models/" + args.model_name + ".pt")
            # wandb.log({"vid": vid})
        elif(log["loss"] < min_loss):
            min_loss = log["loss"]
            best_acmodel = copy.deepcopy(ppo_algo.acmodel)
            torch.save(ppo_algo.acmodel.state_dict(), "models/" + args.model_name + ".pt")
            wandb.save("models/" + args.model_name + ".pt")
        if(i % 5 == 0):
            eval_agent = utils.agent.load_ppo_agent(ppo_algo.acmodel,obss_preprocessor,argmax=False)
            eval_logs = TC_batch_evaluate(eval_agent,args.env,seed=args.seed,episodes=4,return_obss_actions=True,pixel=False)
            
            wandb.log({"eval_return_per_episode": eval_logs["return_per_episode"], "eval_num_frames_per_episode": eval_logs["num_frames_per_episode"]})
            episode_vis = np.random.choice(len(eval_logs["visualization_per_episode"]))
            episode_vis = np.array(eval_logs["visualization_per_episode"][episode_vis])
            vid = wandb.Video(np.transpose(episode_vis, (0, 3, 1, 2)))
            wandb.log({"vid": vid})
        wandb.log({"avg_return_per_episode": sum(log["return_per_episode"]) / len(log["return_per_episode"]), "num_frames": log["num_frames"], "episodes_done": log["episodes_done"], "entropy": log["entropy"], "value": log["value"], "policy_loss": log["policy_loss"], "value_loss": log["value_loss"], "grad_norm": log["grad_norm"], "loss": log["loss"]})
        print("Epoch " + str(i) +"/" + str(args.ppo_updates))
        
    obss_preprocessor.vocab.save("models/" + args.model_name + "/vocab.json")
    wandb.save("models/" + args.model_name + "/vocab.json")
    eval_agent = utils.agent.load_ppo_agent(best_acmodel,obss_preprocessor,argmax=False)
    eval_logs = TC_batch_evaluate(eval_agent,args.env,seed=args.seed,episodes=512,return_obss_actions=True,pixel=False)
    
    wandb.log({"eval_return_per_episode": eval_logs["return_per_episode"], "eval_num_frames_per_episode": eval_logs["num_frames_per_episode"]})
    episode_vis = np.random.choice(len(eval_logs["visualization_per_episode"]))
    episode_vis = np.array(eval_logs["visualization_per_episode"][episode_vis])
    vid = wandb.Video(np.transpose(episode_vis, (0, 3, 1, 2)))
    wandb.log({"vid": vid})
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)