import os
import json
import numpy
import re
import torch
import babyai.rl

from .. import utils


def get_vocab_path(model_name):
    return os.path.join(utils.get_model_dir(model_name), "vocab.json")


class Vocabulary:
    def __init__(self, model_name):
        self.path = get_vocab_path(model_name)
        self.max_size = 100
        self.startToken = 1
        self.endToken = 2
        self.startString = 'sos'
        self.endString = 'eos'

        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))
        else:
            self.vocab = {self.startString : self.startToken, self.endString : self.endToken}

    def __getitem__(self, token):
        if not (token in self.vocab.keys()):
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self, path=None):
        if path is None:
            path = self.path
        utils.create_folders_if_necessary(path)
        json.dump(self.vocab, open(path, "w"))

    def copy_vocab_from(self, other):
        '''
        Copy the vocabulary of another Vocabulary object to the current object.
        '''
        self.vocab.update(other.vocab)


class InstructionsPreprocessor(object):
    def __init__(self, model_name, load_vocab_from=None):
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)

        path = get_vocab_path(model_name)
        if not os.path.exists(path) and load_vocab_from is not None:
            # self.vocab.vocab should be an empty dict
            secondary_path = get_vocab_path(load_vocab_from)
            if os.path.exists(secondary_path):
                old_vocab = Vocabulary(load_vocab_from)
                self.vocab.copy_vocab_from(old_vocab)
            else:
                raise FileNotFoundError('No pre-trained model under the specified name')

    def __call__(self, obss, device=None):
        raw_instrs = []
        max_instr_len = 0

        for obs in obss:
            tokens = re.findall("([a-z]+)", obs["mission"].lower())
            instr = numpy.array([self.vocab[token] for token in tokens])
            raw_instrs.append(instr)
            max_instr_len = max(len(instr), max_instr_len)

        instrs = numpy.zeros((len(obss), max_instr_len))

        for i, instr in enumerate(raw_instrs):
            instrs[i, :len(instr)] = instr

        instrs = torch.tensor(instrs, device=device, dtype=torch.long)
        return instrs
    
class UnifiedLanguagePreprocessor(object):
    def __init__(self, model_name, load_vocab_from=None):
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)

        self.cache = {}

        path = get_vocab_path(model_name)
        if not os.path.exists(path) and load_vocab_from is not None:
            # self.vocab.vocab should be an empty dict
            secondary_path = get_vocab_path(load_vocab_from)
            if os.path.exists(secondary_path):
                old_vocab = Vocabulary(load_vocab_from)
                self.vocab.copy_vocab_from(old_vocab)
            else:
                raise FileNotFoundError('No pre-trained model under the specified name')

    def __call__(self, obss, device=None, train=True):

        # instr
        raw_instrs = []
        max_instr_len = 0

        for obs in obss:
            tokens = re.findall("([a-z]+)", obs['mission'].lower())
            instr = numpy.array([self.vocab[token] for token in tokens])
            self.cache[obs['mission']] = instr
                
            
            raw_instrs.append(instr)
            max_instr_len = max(len(instr), max_instr_len)

        instrs = numpy.zeros((len(obss), max_instr_len))

        for i, instr in enumerate(raw_instrs):
            instrs[i, :len(instr)] = instr

        overall_goals = torch.tensor(instrs, device=device, dtype=torch.long)

        if train:
        # subgoal
            raw_instrs = []
            max_instr_len = 0

            for obs in obss:
                if obs['subgoal'] in self.cache:
                    instr = self.cache[obs['subgoal']]
                else:
                    tokens = re.findall("([a-z]+)", 'sos ' + obs['subgoal'].lower() + ' eos')
                    instr = numpy.array([self.vocab[token] for token in tokens])
                    self.cache[obs['subgoal']] = instr

                raw_instrs.append(instr)
                max_instr_len = max(len(instr), max_instr_len)

            instrs = numpy.zeros((len(obss), max_instr_len))

            for i, instr in enumerate(raw_instrs):
                instrs[i, :len(instr)] = instr

            subgoals = torch.tensor(instrs, device=device, dtype=torch.long)
            max_subgoal_len = max_instr_len

            # history_sg
            complete_flag = numpy.zeros(len(obss))
            history_sgs = []

            for i, obs in enumerate(obss):
                complete_flag[i] = 1 if obs['is_new_sg'] else 0

            complete_flag = torch.tensor(complete_flag, device=device, dtype=torch.long)

            return overall_goals, subgoals, complete_flag
        else:
            return overall_goals, None, None
            
class SubgoalCompletedFlagProprocessor():
    def __call__(self, obss, device=None):
        flag = numpy.array([obs["is_new_sg"] for obs in obss])
        flag = torch.tensor(flag, device=device) #dtype?
        return flag

class RawImagePreprocessor(object):
    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        images = torch.tensor(images, device=device, dtype=torch.float)
        return images


class IntImagePreprocessor(object):
    def __init__(self, num_channels, max_high=255):
        self.num_channels = num_channels
        self.max_high = max_high
        self.offsets = numpy.arange(num_channels) * max_high
        self.max_size = int(num_channels * max_high)

    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        # The padding index is 0 for all the channels
        images = (images + self.offsets) * (images > 0)
        images = torch.tensor(images, device=device, dtype=torch.long)
        return images

class TCObssPreprocessor:
    def __init__(self, model_name, obs_space=None, load_vocab_from=None):
        self.image_preproc = RawImagePreprocessor()
        self.language_preproc = UnifiedLanguagePreprocessor(model_name, load_vocab_from)
        self.vocab = self.language_preproc.vocab
        self.obs_space = {
            "image": 147,
            "instr": self.vocab.max_size
        }
        self.index2word = {index:word for word,index in self.vocab.vocab.items()}

    def __call__(self, obss, device=None, train=True):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        obs_.instr, obs_.subgoal, obs_.is_new_sg = self.language_preproc(obss, device=device, train=train)

        return obs_
    
    def decode_subgoal(self, subgoals):
        subgoal_sentences = []
        for idx in range(subgoals.size(0)):
            sg = subgoals[idx, :]
            sg_sentence = []
            for token in sg:
                token = token.item()
                if token == self.vocab.startToken:
                    continue
                elif token == self.vocab.endToken:
                    break
                else:
                    sg_sentence.append(self.index2word[token])
            subgoal_sentences.append(" ".join(sg_sentence))
        
        return subgoal_sentences
    
    def encode_subgoal(self, subgoal):
        tokens = re.findall("([a-z]+)", 'sos ' + subgoal.lower() + ' eos')
        instr = numpy.array([self.vocab[token] for token in tokens])
        return instr

class ObssPreprocessor:
    def __init__(self, model_name, obs_space=None, load_vocab_from=None):
        self.image_preproc = RawImagePreprocessor()
        self.instr_preproc = InstructionsPreprocessor(model_name, load_vocab_from)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": 147,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_


class IntObssPreprocessor(object):
    def __init__(self, model_name, obs_space, load_vocab_from=None):
        image_obs_space = obs_space.spaces["image"]
        self.image_preproc = IntImagePreprocessor(image_obs_space.shape[-1],
                                                  max_high=image_obs_space.high.max())
        self.instr_preproc = InstructionsPreprocessor(load_vocab_from or model_name)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": self.image_preproc.max_size,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_
