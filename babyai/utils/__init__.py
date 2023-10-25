import os
import random
import numpy
import torch
from babyai.utils.agent import load_agent, ModelAgent, DemoAgent, BotAgent
from babyai.utils.demos import (
    load_demos,
    save_demos,
    synthesize_demos,
    get_demos_path,
    get_subgoal_sentence,
)
from babyai.utils.format import (
    ObssPreprocessor,
    IntObssPreprocessor,
    get_vocab_path,
    TCObssPreprocessor,
)
from babyai.utils.log import get_log_path, get_log_dir, synthesize, configure_logging
from babyai.utils.model import get_model_dir, load_model, save_model

group_name = ""


def expr_group_name():
    return group_name


def storage_dir():
    # defines manually here
    return "."
    # could also difine from environment
    # return os.environ.get("TC_STORAGE", '.')


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not (os.path.isdir(dirname)):
        os.makedirs(dirname)


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
