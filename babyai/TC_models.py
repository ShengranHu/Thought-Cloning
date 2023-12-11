import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import weight_norm

import babyai.rl
from babyai.model import FiLM, ImageBOWEmbedding, initialize_parameters
from babyai.submodules import REFER, ScaledDotProductAttention, Decoder

import pdb


class ObsEmbedding(nn.Module):
    def __init__(self, obs_space, image_dim, instr_dim):
        super().__init__()

        self.image_dim = image_dim
        self.instr_dim = instr_dim

        self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)

        self.image_conv = nn.Sequential(
            *[
                *([ImageBOWEmbedding(obs_space["image"], 128)]),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ]
        )

        gru_dim = self.instr_dim
        gru_dim //= 2

        self.instr_rnn = nn.GRU(
            self.instr_dim, gru_dim, batch_first=True, bidirectional=True
        )
        self.subgoal_rnn = nn.GRU(
            self.instr_dim, gru_dim, batch_first=True, bidirectional=True
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def get_visual_embedding(self, x):
        x = torch.transpose(torch.transpose(x, 1, 3), 2, 3)
        x = self.image_conv(x)
        return x

    def get_language_embedding(self, instr, is_subgoal):
        lengths = (instr != 0).sum(1).long()

        masks = (instr != 0).float()

        if lengths.shape[0] > 1:
            seq_lengths, perm_idx = lengths.sort(0, descending=True)
            iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
            if instr.is_cuda:
                iperm_idx = iperm_idx.cuda()
            for i, v in enumerate(perm_idx):
                iperm_idx[v.data] = i

            inputs = self.word_embedding(instr)
            inputs = inputs[perm_idx]

            inputs = pack_padded_sequence(
                inputs, seq_lengths.data.cpu().numpy(), batch_first=True
            )

            if is_subgoal:
                outputs, final_states = self.subgoal_rnn(inputs)
            else:
                outputs, final_states = self.instr_rnn(inputs)
        else:
            instr = instr[:, 0 : lengths[0]]
            if is_subgoal:
                outputs, final_states = self.subgoal_rnn(self.word_embedding(instr))
            else:
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
            iperm_idx = None

        final_states = final_states.transpose(0, 1).contiguous()
        final_states = final_states.view(final_states.shape[0], -1)
        if iperm_idx is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[iperm_idx]
            final_states = final_states[iperm_idx]

        return outputs


class UpperLevel(nn.Module):
    def __init__(
        self, obs_space, image_dim, memory_dim, final_instr_dim, embedding_layer
    ):
        super().__init__()

        self.obs_space = obs_space
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.final_instr_dim = final_instr_dim

        # encode subgoals

        self.history_rnn = nn.LSTMCell(self.final_instr_dim, self.memory_dim)
        self.history2key = nn.Linear(self.memory_size, self.final_instr_dim)

        # attn unit
        self.attention = ScaledDotProductAttention(
            temperature=np.power(self.final_instr_dim, 0.5)
        )

        # memory mapping
        self.memory2key_instr = nn.Linear(self.memory_size, self.final_instr_dim)
        self.memory2key_sg = nn.Linear(self.memory_size, self.final_instr_dim)

        # main branch
        self.instr_historysg_attn = nn.ModuleList(
            [
                REFER(
                    d_model=self.final_instr_dim,
                    d_inner=self.final_instr_dim,
                    n_head=2,
                    d_k=self.final_instr_dim // 2,
                    d_v=self.final_instr_dim // 2,
                    dropout=0.2,
                )
                for _ in range(2)
            ]
        )

        # modality fusion
        num_module = 2
        self.controllers = []
        for ni in range(num_module):
            mod = FiLM(
                in_features=self.final_instr_dim,
                out_features=128 if ni < num_module - 1 else self.image_dim,
                in_channels=128,
                imm_channels=128,
            )
            self.controllers.append(mod)
            self.add_module("FiLM_" + str(ni), mod)

        self.film_pool = nn.MaxPool2d(kernel_size=(7, 7), stride=2)

        # memory unit
        self.numLayers = 1
        self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)
        self.embedding_size = self.semi_memory_size

        # decoder
        self.decoder = Decoder(
            obs_space["instr"],
            self.final_instr_dim,
            embedding_layer,
            self.memory_dim,
            numLayers=self.numLayers,
        )
        self.MAX_SEQ_LEN = 15

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forwardDecode(
        self,
        visual_embedding,
        memory,
        overall_goal_embedding,
        overall_goal,
        subgoal_histories,
        beamSize=1,
    ):
        mask = (overall_goal == 0).unsqueeze(1)
        mask = mask[:, :, : overall_goal_embedding.shape[1]]
        overall_goal_embedding = overall_goal_embedding[:, : mask.shape[2], :]

        keys = self.memory2key_instr(memory).unsqueeze(1)
        overall_goal_embedding_attn, _ = self.attention(
            keys, overall_goal_embedding, overall_goal_embedding, mask
        )

        # transformer encoder layers (subgoal_history, overall_goal)
        context = self.history2key(subgoal_histories).unsqueeze(1)
        for enc_layer in self.instr_historysg_attn:
            context, _ = enc_layer(context, overall_goal_embedding)

        context = (context + overall_goal_embedding_attn).squeeze(1)

        # modality fusion
        x = visual_embedding
        for controller in self.controllers:
            out = controller(x, context)
            x = x + out
        x = F.relu(self.film_pool(x))
        x = x.reshape(x.shape[0], -1)

        # upper level memory
        hidden = (
            memory[:, : self.semi_memory_size],
            memory[:, self.semi_memory_size :],
        )
        hidden = self.memory_rnn(x, hidden)
        memory = torch.cat(hidden, dim=1)

        sg, logProbs = self.decoder.forwardDecode(
            hidden, maxSeqLen=self.MAX_SEQ_LEN, inference="greedy", beamSize=beamSize
        )
        return sg, memory

    def forward(
        self,
        visual_embedding,
        memory,
        overall_goal_embedding,
        overall_goal,
        subgoal_histories,
        teacher_forcing=True,
        gt_subgoal=None,
        gt_subgoal_embedding=None,
    ):
        """
        visual_embedding: [batch, image_dim]
        memory: [batch, memory_dim]

        overall_goal_embedding: [batch, word_len, final_instr_dim]
        overall_goal: [batch, word_len, vocab_size]

        subgoal_histories: [batch, memory_size]
            Already get attn with memory in prev timestep
            None if first time step

        gt_subgoal: for teacher forcing training, [batch, word_len, vocab_size]
        gt_subgoal_embedding: for teacher forcing training, [batch, word_len, final_instr_dim]
        """

        # next subgoal generation branch

        # attn overall goal with memory
        mask = (overall_goal == 0).unsqueeze(1)
        mask = mask[:, :, : overall_goal_embedding.shape[1]]
        overall_goal_embedding = overall_goal_embedding[:, : mask.shape[2], :]

        keys = self.memory2key_instr(memory).unsqueeze(1)
        overall_goal_embedding_attn, _ = self.attention(
            keys, overall_goal_embedding, overall_goal_embedding, mask
        )

        # transformer encoder layers (subgoal_history, overall_goal)
        context = self.history2key(subgoal_histories).unsqueeze(1)
        for enc_layer in self.instr_historysg_attn:
            context, _ = enc_layer(context, overall_goal_embedding)

        context = (context + overall_goal_embedding_attn).squeeze(1)

        # modality fusion
        x = visual_embedding
        for controller in self.controllers:
            out = controller(x, context)
            x = x + out
        x = F.relu(self.film_pool(x))
        x = x.reshape(x.shape[0], -1)

        # upper level memory
        hidden = (
            memory[:, : self.semi_memory_size],
            memory[:, self.semi_memory_size :],
        )
        hidden = self.memory_rnn(x, hidden)
        memory = torch.cat(hidden, dim=1)

        # decode
        if teacher_forcing:
            logProbs = self.decoder.forward(hidden, gt_subgoal)

            return logProbs, gt_subgoal, memory
        else:
            sg, logProbs = self.decoder.forwardDecode(
                hidden, maxSeqLen=self.MAX_SEQ_LEN
            )
            return logProbs, sg, memory


class LowerLevel(nn.Module):
    def __init__(self, action_space, image_dim, memory_dim, final_instr_dim):
        super().__init__()

        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.final_instr_dim = final_instr_dim

        # attn unit
        self.attention = ScaledDotProductAttention(
            temperature=np.power(self.final_instr_dim, 0.5)
        )

        self.memory2key_sg = nn.Linear(self.memory_size, self.final_instr_dim)
        self.memory2key_instr = nn.Linear(self.memory_size, self.final_instr_dim)

        self.sg_linear = nn.Linear(self.final_instr_dim, self.final_instr_dim)
        self.instr_linear = nn.Linear(self.final_instr_dim, self.final_instr_dim)

        num_module = 2
        self.controllers = []
        for ni in range(num_module):
            mod = FiLM(
                in_features=self.final_instr_dim,
                out_features=128 if ni < num_module - 1 else self.image_dim,
                in_channels=128,
                imm_channels=128,
            )
            self.controllers.append(mod)
            self.add_module("FiLM_" + str(ni), mod)

        self.film_pool = nn.MaxPool2d(kernel_size=(7, 7), stride=2)

        # Define memory and resize image embedding
        self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)
        self.embedding_size = self.semi_memory_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, action_space.n)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(
        self,
        visual_embedding,
        memory,
        subgoal,
        subgoal_embedding,
        instr,
        instr_embedding,
    ):
        # atten subgoal with memory
        mask = (subgoal == 0).unsqueeze(1)
        mask = mask[:, :, : subgoal_embedding.shape[1]]
        subgoal_embedding = subgoal_embedding[:, : mask.shape[2], :]

        keys = self.memory2key_sg(memory).unsqueeze(1)
        subgoal_embedding, attn = self.attention(
            keys, subgoal_embedding, subgoal_embedding, mask
        )

        # atten instr with memory
        mask = (instr == 0).unsqueeze(1)
        mask = mask[:, :, : instr_embedding.shape[1]]
        instr_embedding = instr_embedding[:, : mask.shape[2], :]

        keys = self.memory2key_instr(memory).unsqueeze(1)
        instr_embedding, attn = self.attention(
            keys, instr_embedding, instr_embedding, mask
        )

        language_embedding = (
            self.sg_linear(subgoal_embedding) + self.instr_linear(instr_embedding)
        ).squeeze(1)

        x = visual_embedding

        for controller in self.controllers:
            out = controller(x, language_embedding)
            # residual
            x = x + out

        x = F.relu(self.film_pool(x))
        x = x.reshape(x.shape[0], -1)

        hidden = (
            memory[:, : self.semi_memory_size],
            memory[:, self.semi_memory_size :],
        )
        hidden = self.memory_rnn(x, hidden)
        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)

        y = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(y, dim=1))

        z = self.critic(embedding)
        value = z.squeeze(1)

        return dist, value, memory


class ThoughCloningModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(
        self, obs_space, action_space, image_dim=128, memory_dim=2048, instr_dim=256
    ):
        super().__init__()

        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        self.obs_space = obs_space

        # define observation embedding module

        self.obs_embedding = ObsEmbedding(obs_space, image_dim, instr_dim)
        self.final_instr_dim = self.instr_dim

        # define upper_level_policy

        self.upper_level_policy = UpperLevel(
            obs_space,
            image_dim,
            memory_dim,
            self.final_instr_dim,
            self.obs_embedding.word_embedding,
        )

        # define lower_level_policy

        self.lower_level_policy = LowerLevel(
            action_space, image_dim, memory_dim, self.final_instr_dim
        )

        self.control_linear = nn.Linear(self.obs_space["instr"], instr_dim)
        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def _get_instr_embedding(self, instr):
        return self.obs_embedding.get_language_embedding(instr, is_subgoal=False)

    def train_forward(self, obs, memory, teacher_forcing_ratio, instr_embedding=None):
        """
        obs.instr: overall goal
        obs.image: observation
        memory: [batch, 3, memory_dim] (upper, lower, history memory)


        instr_embedding: [batch, sentence_len, instr_dim] (saved for computational efficiency)
        """

        # encoding
        if instr_embedding == None:
            instr_embedding = self._get_instr_embedding(obs.instr)

        gt_subgoal = obs.subgoal
        gt_subgoal_embedding = self.obs_embedding.get_language_embedding(
            gt_subgoal, is_subgoal=True
        )
        visual_embedding = self.obs_embedding.get_visual_embedding(obs.image)

        # memory handle
        upper_memory = memory[:, 0, :]
        lower_memory = memory[:, 1, :]
        subgoal_histories = memory[:, 2, :]

        teacher_forcing = np.random.random() < teacher_forcing_ratio

        if teacher_forcing:
            # upper level
            logProbs, subgoal, upper_memory = self.upper_level_policy.forward(
                visual_embedding,
                upper_memory,
                instr_embedding,
                obs.instr,
                subgoal_histories,
                teacher_forcing=teacher_forcing,
                gt_subgoal=gt_subgoal,
                gt_subgoal_embedding=gt_subgoal_embedding,
            )
            # attn subgoal with new memory
            mask = (gt_subgoal == 0).unsqueeze(1)
            mask = mask[:, :, : gt_subgoal_embedding.shape[1]]
            gt_subgoal_embedding = gt_subgoal_embedding[:, : mask.shape[2], :]

            keys = self.upper_level_policy.memory2key_sg(upper_memory).unsqueeze(1)
            next_subgoal_history, _ = self.upper_level_policy.attention(
                keys, gt_subgoal_embedding, gt_subgoal_embedding, mask
            )

            # update subgoal_histories
            hidden = (
                subgoal_histories[:, : self.semi_memory_size],
                subgoal_histories[:, self.semi_memory_size :],
            )
            hidden = self.upper_level_policy.history_rnn(
                next_subgoal_history.squeeze(1), hidden
            )
            subgoal_histories = torch.cat(hidden, dim=1)

            # lower level
            dist, value, lower_memory = self.lower_level_policy.forward(
                visual_embedding,
                lower_memory,
                gt_subgoal,
                gt_subgoal_embedding,
                obs.instr,
                instr_embedding,
            )

            # handle memory
            memory = torch.stack((upper_memory, lower_memory, subgoal_histories), dim=1)

            return {
                "memory": memory,
                "dist": dist,
                "logProbs": logProbs,
                "subgoal": gt_subgoal,
                "predicted_subgoal": subgoal,
            }
        else:
            logProbs, subgoal, upper_memory = self.upper_level_policy.forward(
                visual_embedding,
                upper_memory,
                instr_embedding,
                obs.instr,
                subgoal_histories,
                teacher_forcing=teacher_forcing,
                gt_subgoal=gt_subgoal,
                gt_subgoal_embedding=gt_subgoal_embedding,
            )
            # encode subgoal
            subgoal_embedding = self.obs_embedding.get_language_embedding(
                subgoal, is_subgoal=True
            )

            # attn subgoal with new memory
            mask = (subgoal == 0).unsqueeze(1)
            mask = mask[:, :, : subgoal_embedding.shape[1]]
            subgoal_embedding = subgoal_embedding[:, : mask.shape[2], :]

            keys = self.upper_level_policy.memory2key_sg(upper_memory).unsqueeze(1)
            next_subgoal_history, _ = self.upper_level_policy.attention(
                keys, subgoal_embedding, subgoal_embedding, mask
            )

            # update subgoal_histories
            hidden = (
                subgoal_histories[:, : self.semi_memory_size],
                subgoal_histories[:, self.semi_memory_size :],
            )
            hidden = self.upper_level_policy.history_rnn(
                next_subgoal_history.squeeze(1), hidden
            )
            subgoal_histories = torch.cat(hidden, dim=1)

            # lower level
            dist, value, lower_memory = self.lower_level_policy.forward(
                visual_embedding,
                lower_memory,
                subgoal,
                subgoal_embedding,
                obs.instr,
                instr_embedding,
            )

            # handle memory
            memory = torch.stack((upper_memory, lower_memory, subgoal_histories), dim=1)

            return {
                "memory": memory,
                "dist": dist,
                "logProbs": logProbs,
                "subgoal": gt_subgoal,
                "predicted_subgoal": subgoal,
            }

    def forward(self, obs, memory, instr_embedding=None, beamSize=1):
        if instr_embedding == None:
            instr_embedding = self._get_instr_embedding(obs.instr)

        visual_embedding = self.obs_embedding.get_visual_embedding(obs.image)

        # memory handle
        upper_memory = memory[:, 0, :]
        lower_memory = memory[:, 1, :]
        subgoal_histories = memory[:, 2, :]

        subgoal, upper_memory = self.upper_level_policy.forwardDecode(
            visual_embedding,
            upper_memory,
            instr_embedding,
            obs.instr,
            subgoal_histories,
            beamSize=beamSize,
        )

        # encode subgoal
        subgoal_embedding = self.obs_embedding.get_language_embedding(
            subgoal, is_subgoal=True
        )

        # attn subgoal with new memory
        mask = (subgoal == 0).unsqueeze(1)
        mask = mask[:, :, : subgoal_embedding.shape[1]]
        subgoal_embedding = subgoal_embedding[:, : mask.shape[2], :]

        keys = self.upper_level_policy.memory2key_sg(upper_memory).unsqueeze(1)
        next_subgoal_history, _ = self.upper_level_policy.attention(
            keys, subgoal_embedding, subgoal_embedding, mask
        )

        # update subgoal_histories
        hidden = (
            subgoal_histories[:, : self.semi_memory_size],
            subgoal_histories[:, self.semi_memory_size :],
        )
        hidden = self.upper_level_policy.history_rnn(
            next_subgoal_history.squeeze(1), hidden
        )
        subgoal_histories = torch.cat(hidden, dim=1)

        # lower level
        dist, value, lower_memory = self.lower_level_policy.forward(
            visual_embedding,
            lower_memory,
            subgoal,
            subgoal_embedding,
            obs.instr,
            instr_embedding,
        )

        # handle memory
        memory = torch.stack((upper_memory, lower_memory, subgoal_histories), dim=1)

        return {
            "memory": memory,
            "dist": dist,
            "subgoal": subgoal,
            "value": value,
            "extra_predictions": dict(),
        }

    def rl_forward(self, obs, memory, instr_embedding=None, lower_only=False):
        if instr_embedding == None:
            instr_embedding = self._get_instr_embedding(obs.instr)

        gt_subgoal = obs.subgoal
        gt_subgoal_embedding = self.obs_embedding.get_language_embedding(
            gt_subgoal, is_subgoal=True
        )
        visual_embedding = self.obs_embedding.get_visual_embedding(obs.image)

        # memory handle
        upper_memory = memory[:, 0, :]
        lower_memory = memory[:, 1, :]
        subgoal_histories = memory[:, 2, :]
        logProbs, subgoal, upper_memory = self.upper_level_policy.forward(
            visual_embedding,
            upper_memory,
            instr_embedding,
            obs.instr,
            subgoal_histories,
            teacher_forcing=0,
            gt_subgoal=gt_subgoal,
            gt_subgoal_embedding=gt_subgoal_embedding,
        )
        # encode subgoal
        subgoal_embedding = self.obs_embedding.get_language_embedding(
            subgoal, is_subgoal=True
        )

        # attn subgoal with new memory
        mask = (subgoal == 0).unsqueeze(1)
        mask = mask[:, :, : subgoal_embedding.shape[1]]
        subgoal_embedding = subgoal_embedding[:, : mask.shape[2], :]

        keys = self.upper_level_policy.memory2key_sg(upper_memory).unsqueeze(1)
        next_subgoal_history, _ = self.upper_level_policy.attention(
            keys, subgoal_embedding, subgoal_embedding, mask
        )

        # update subgoal_histories
        hidden = (
            subgoal_histories[:, : self.semi_memory_size],
            subgoal_histories[:, self.semi_memory_size :],
        )
        hidden = self.upper_level_policy.history_rnn(
            next_subgoal_history.squeeze(1), hidden
        )
        subgoal_histories = torch.cat(hidden, dim=1)
        if lower_only:
            visual_embedding = visual_embedding.detach().clone()
            lower_memory = lower_memory.detach().clone()
            subgoal = subgoal.detach().clone()
            subgoal_embedding = subgoal_embedding.detach().clone()
            obs.instr = obs.instr.detach().clone()
            instr_embedding = instr_embedding.detach().clone()

        # lower level
        dist, value, lower_memory = self.lower_level_policy.forward(
            visual_embedding,
            lower_memory,
            subgoal,
            subgoal_embedding,
            obs.instr,
            instr_embedding,
        )

        # handle memory
        memory = torch.stack((upper_memory, lower_memory, subgoal_histories), dim=1)

        # pdb.set_trace()
        return {
            "memory": memory,
            "dist": dist,
            "subgoal": gt_subgoal,
            "predicted_subgoal": subgoal,
            "value": value,
            "logProbs": logProbs,
        }
