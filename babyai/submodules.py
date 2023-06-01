""" 
This code is modified from Gi-Cheon Kang's repository
https://github.com/gicheonkang/dan-visdial
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

class REFER(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.2):
        super(REFER, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, q, m):
        enc_output, enc_slf_attn = self.slf_attn(q, m, m)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        q: [batch, 1, 512]
        k, v: [batch, num_entry, 512]
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class LayerNorm(nn.Module):
    """ 
    Layer Normalization 
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta



### decoder part 
"""This code is modified from 
https://github.com/batra-mlp-lab/visdial-rl
and 
https://github.com/mcogswell/dialog_without_dialog/blob/master/models/decoder.py""" 

def maskedNll(seq, gtSeq, returnScores=False):
    '''
    Compute the NLL loss of ground truth (target) sentence given the
    model. Assumes that gtSeq has <START> and <END> token surrounding
    every sequence and gtSeq is left aligned (i.e. right padded)
    S: <START>, E: <END>, W: word token, 0: padding token, P(*): logProb
        gtSeq:
            [ S     W1    W2  E   0   0]
        Teacher forced logProbs (seq):
            [P(W1) P(W2) P(E) -   -   -]
        Required gtSeq (target):
            [  W1    W2    E  0   0   0]
        Mask (non-zero tokens in target):
            [  1     1     1  0   0   0]
    '''
    # Shifting gtSeq 1 token left to remove <START>
    padColumn = gtSeq.data.new(gtSeq.size(0), 1).fill_(0)
    padColumn = Variable(padColumn)
    target = torch.cat([gtSeq, padColumn], dim=1)[:, 1:]

    # Generate a mask of non-padding (non-zero) tokens
    mask = target.data.gt(0)
    loss = 0
    if isinstance(gtSeq, Variable):
        mask = Variable(mask, volatile=gtSeq.volatile)
    assert isinstance(target, Variable)
    gtLogProbs = torch.gather(seq, 2, target.unsqueeze(2)).squeeze(2)
    # Mean sentence probs:
    # gtLogProbs = gtLogProbs/(mask.float().sum(1).view(-1,1))
    if returnScores:
        return (gtLogProbs * (mask.float())).sum(1)
    maskedLL = torch.masked_select(gtLogProbs, mask)
    nll_loss = -torch.sum(maskedLL) / seq.size(0)
    return nll_loss

infty = float('inf')

class Decoder(nn.Module):
    def __init__(self,
                 vocabSize,
                 embedSize,
                 wordEmbed,
                 rnnHiddenSize,
                 numLayers=2,
                 dropout=0,
                 **kwargs):
        super(Decoder, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers
        self.dropout = dropout
        self.wordEmbed = wordEmbed

        self.startToken = 1
        self.endToken = 2

        # Modules
        self.rnn = nn.LSTM(
            self.embedSize,
            self.rnnHiddenSize,
            self.numLayers,
            batch_first=True,
            dropout=self.dropout)
        self.outNet = nn.Linear(self.rnnHiddenSize, self.vocabSize)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, encStates, inputSeq):
        '''
        Given encoder states, forward pass an input sequence 'inputSeq' to
        compute its log likelihood under the current decoder RNN state.
        Arguments:
            encStates: (H, C) Tuple of hidden and cell encoder states
            inputSeq: Input sequence for computing log probabilities
        Output:
            A (batchSize, length, vocabSize) sized tensor of log-probabilities
            obtained from feeding 'inputSeq' to decoder RNN at evert time step
        Note:
            Maximizing the NLL of an input sequence involves feeding as input
            tokens from the GT (ground truth) sequence at every time step and
            maximizing the probability of the next token ("teacher forcing").
            See 'maskedNll' in utils/utilities.py where the log probability of
            the next time step token is indexed out for computing NLL loss.
        '''
        if inputSeq is not None:
            inputSeq = self.wordEmbed(inputSeq)
            encStates = (encStates[0].unsqueeze(0), encStates[1].unsqueeze(0))
            outputs, _ = self.rnn(inputSeq, encStates)
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputSize = outputs.size()
            flatOutputs = outputs.reshape(-1, outputSize[2])
            flatScores = self.outNet(flatOutputs)
            flatLogProbs = self.logSoftmax(flatScores)
            logProbs = flatLogProbs.view(outputSize[0], outputSize[1], -1)
        return logProbs

    def forwardDecode(self,
                      encStates,
                      maxSeqLen=20,
                      inference='greedy',
                      beamSize=1):
        '''
        Decode a sequence of tokens given an encoder state, using either
        sampling or greedy inference.
        Arguments:
            encStates : (H, C) Tuple of hidden and cell encoder states
            maxSeqLen : Maximum length of token sequence to generate
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            beamSize  : Beam search width
        Notes:
            * 
            * Greedy inference is used for evaluation
            * 
        '''
        # if inference == 'greedy' and beamSize > 1:
        #     # Use beam search inference when beam size is > 1
        #     return self.beamSearchDecoder(encStates, beamSize, maxSeqLen)
        
        # handle encState
        hid = (encStates[0].unsqueeze(0), encStates[1].unsqueeze(0))
        
        if self.outNet.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch

        self.samples = []
        maxLen = maxSeqLen + 1  # Extra <END> token
        self.batchSize = batchSize = hid[0].size(1)
        # Placeholder for filling in tokens at evert time step
        seq = th.LongTensor(batchSize, maxLen + 1)
        seq.fill_(self.endToken)
        seq[:, 0] = self.startToken
        seq = Variable(seq, requires_grad=False)

        sampleLens = th.LongTensor(batchSize).fill_(0)
        # Tensors needed for tracking sampleLens
        unitColumn = th.LongTensor(batchSize).fill_(1)
        mask = th.BoolTensor(seq.size()).fill_(0)

        saved_log_probs = []

        # Generating tokens sequentially
        for t in range(maxLen - 1):
            emb = self.wordEmbed(seq[:, t:t + 1])
            # emb has shape  (batch, 1, embedSize)
            output, hid = self.rnn(emb, hid)
            # output has shape (batch, 1, rnnHiddenSize)
            scores = self.outNet(output.squeeze(1))

            # Explicitly removing padding token (index 0) and <START> token
            # (index -2) from scores so that they are never sampled.
            # This is allows us to keep <START> and padding token in
            # the decoder vocab without any problems in RL sampling.
            # if t > 0:
            #     scores[:, 0] = -infty
            #     scores[:, self.startToken] = -infty
            # elif t == 0:
            #     # Additionally, remove <END> token from the first sample
            #     # to prevent the sampling of an empty sequence.
            #     scores[:, 0] = -infty
            #     scores[:, self.startToken] = -infty
            #     scores[:, self.endToken] = -infty

            if inference == 'sample':
                logProb = self.logSoftmax(scores)
                probs = torch.exp(logProb)
                categorical_dist = Categorical(probs)
                sample = categorical_dist.sample()
                # Saving log probs for a subsequent reinforce call
                sample = sample.unsqueeze(-1)
            # elif inference == 'greedy' and beamSize > 1 and t == 0:
            #     logProb = self.logSoftmax(scores)
            #     _, sample = torch.topk(logProb, 2)
            #     sample = sample[:, -1].reshape(-1, 1)

            elif inference == 'greedy':
                logProb = self.logSoftmax(scores)
                _, sample = torch.max(logProb, dim=1, keepdim=True)
            else:
                raise ValueError(
                    "Invalid inference type: '{}'".format(inference))

            self.samples.append(sample)
            saved_log_probs.append(logProb)

            seq.data[:, t + 1] = sample.data.squeeze(1)
            # Marking spots where <END> token is generated
            mask[:, t] = sample.data.eq(self.endToken).squeeze(1)

        mask[:, maxLen - 1].fill_(1)

        # Computing lengths of generated sequences
        for t in range(maxLen):
            # Zero out the spots where end token is reached
            unitColumn.masked_fill_(mask[:, t], 0)
            # Update mask
            mask[:, t] = unitColumn
            # Add +1 length to all un-ended sequences
            sampleLens = sampleLens + unitColumn

        # Keep mask for later use in RL reward masking
        self.mask = Variable(mask, requires_grad=False)

        # Adding <START> and <END> to generated answer lengths for consistency
        maxLenCol = torch.zeros_like(sampleLens) + maxLen
        sampleLens = torch.min(sampleLens + 1, maxLenCol)
        sampleLens = Variable(sampleLens, requires_grad=False)

        startColumn = sample.data.new(sample.size()).fill_(self.startToken)
        startColumn = Variable(startColumn, requires_grad=False)

        # Note that we do not add startColumn to self.samples itself
        # as reinforce is called on self.samples (which needs to be
        # the output of a stochastic function)
        gen_samples = [startColumn] + self.samples

        samples = torch.cat(gen_samples, 1)

        fill_mask = th.BoolTensor(samples.size()).fill_(1)
        fill_mask[:,2:] = mask[:,:samples.size(1)-2]

        samples.masked_fill_(~fill_mask, 0)
        logProbs = torch.stack(saved_log_probs, 1)

        return samples, logProbs

    def beamSearchDecoder(self, initStates, beamSize, maxSeqLen):
        '''
        Beam search for sequence generation
        Arguments:
            initStates - Initial encoder states tuple
            beamSize - Beam Size
            maxSeqLen - Maximum length of sequence to decode
        '''

        # For now, use beam search for evaluation only
        assert self.training == False

        hid = (initStates[0].unsqueeze(0), initStates[1].unsqueeze(0))

        # Determine if cuda tensors are being used
        if self.wordEmbed.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch

        LENGTH_NORM = True
        maxLen = maxSeqLen + 1  # Extra <END> token
        self.batchSize = batchSize = hid[0].size(1)

        startTokenArray = th.LongTensor(batchSize, 1).fill_(self.startToken)
        backVector = th.LongTensor(beamSize)
        torch.arange(0, beamSize, out=backVector)
        backVector = backVector.unsqueeze(0).repeat(batchSize, 1)

        tokenArange = th.LongTensor(self.vocabSize)
        torch.arange(0, self.vocabSize, out=tokenArange)
        tokenArange = Variable(tokenArange)

        startTokenArray = Variable(startTokenArray)
        backVector = Variable(backVector)
        hiddenStates = hid

        # Inits
        beamTokensTable = th.LongTensor(batchSize, beamSize, maxLen).fill_(
            self.endToken)
        beamTokensTable = Variable(beamTokensTable)
        backIndices = th.LongTensor(batchSize, beamSize, maxLen).fill_(-1)
        backIndices = Variable(backIndices)

        aliveVector = beamTokensTable[:, :, 0].eq(self.endToken).unsqueeze(2)

        for t in range(maxLen - 1):  # Beam expansion till maxLen]
            if t == 0:
                # First column of beamTokensTable is generated from <START> token
                emb = self.wordEmbed(startTokenArray)
                # emb has shape (batchSize, 1, embedSize)
                output, hiddenStates = self.rnn(emb, hiddenStates)
                # output has shape (batchSize, 1, rnnHiddenSize)
                scores = self.outNet(output.squeeze(1))
                logProbs = self.logSoftmax(scores)
                # scores & logProbs has shape (batchSize, vocabSize)

                # Find top beamSize logProbs
                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)
                beamTokensTable[:, :, 0] = topIdx.data
                logProbSums = topLogProbs

                # Repeating hiddenStates 'beamSize' times for subsequent self.rnn calls
                hiddenStates = [
                    x.unsqueeze(2).repeat(1, 1, beamSize, 1)
                    for x in hiddenStates
                ]
                hiddenStates = [
                    x.view(self.numLayers, -1, self.rnnHiddenSize)
                    for x in hiddenStates
                ]
                # H_0 and C_0 have shape (numLayers, batchSize*beamSize, rnnHiddenSize)
            else:
                # Subsequent columns are generated from previous tokens
                emb = self.wordEmbed(beamTokensTable[:, :, t - 1])
                # emb has shape (batchSize, beamSize, embedSize)
                output, hiddenStates = self.rnn(
                    emb.view(-1, 1, self.embedSize), hiddenStates)
                # output has shape (batchSize*beamSize, 1, rnnHiddenSize)
                scores = self.outNet(output.squeeze())
                logProbsCurrent = self.logSoftmax(scores)
                # logProbs has shape (batchSize*beamSize, vocabSize)
                # NOTE: Padding token has been removed from generator output during
                # sampling (RL fine-tuning). However, the padding token is still
                # present in the generator vocab and needs to be handled in this
                # beam search function. This will be supported in a future release.
                logProbsCurrent = logProbsCurrent.view(batchSize, beamSize,
                                                    self.vocabSize)

                if LENGTH_NORM:
                    # Add (current log probs / (t+1))
                    logProbs = logProbsCurrent * (aliveVector.float() /
                                                (t + 1))
                    # Add (previous log probs * (t/t+1) ) <- Mean update
                    coeff_ = aliveVector.eq(0).float() + (
                        aliveVector.float() * t / (t + 1))
                    logProbs += logProbSums.unsqueeze(2) * coeff_
                else:
                    # Add currrent token logProbs for alive beams only
                    logProbs = logProbsCurrent * (aliveVector.float())
                    # Add previous logProbSums upto t-1
                    logProbs += logProbSums.unsqueeze(2)

                # Masking out along |V| dimension those sequence logProbs
                # which correspond to ended beams so as to only compare
                # one copy when sorting logProbs
                mask_ = aliveVector.eq(0).repeat(1, 1, self.vocabSize)
                mask_[:, :,
                    0] = 0  # Zeroing all except first row for ended beams
                minus_infinity_ = torch.min(logProbs)
                logProbs.data.masked_fill_(mask_.data, minus_infinity_)

                logProbs = logProbs.view(batchSize, -1)
                tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).\
                                repeat(batchSize,beamSize,1)
                tokensArray.masked_fill_(aliveVector.eq(0), self.endToken)
                tokensArray = tokensArray.view(batchSize, -1)
                backIndexArray = backVector.unsqueeze(2).\
                                repeat(1,1,self.vocabSize).view(batchSize,-1)

                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)

                logProbSums = topLogProbs
                beamTokensTable[:, :, t] = tokensArray.gather(1, topIdx)
                backIndices[:, :, t] = backIndexArray.gather(1, topIdx)

                # Update corresponding hidden and cell states for next time step
                hiddenCurrent, cellCurrent = hiddenStates

                # Reshape to get explicit beamSize dim
                original_state_size = hiddenCurrent.size()
                num_layers, _, rnnHiddenSize = original_state_size
                hiddenCurrent = hiddenCurrent.view(
                    num_layers, batchSize, beamSize, rnnHiddenSize)
                cellCurrent = cellCurrent.view(
                    num_layers, batchSize, beamSize, rnnHiddenSize)

                # Update states according to the next top beams
                backIndexVector = backIndices[:, :, t].unsqueeze(0)\
                    .unsqueeze(-1).repeat(num_layers, 1, 1, rnnHiddenSize)
                hiddenCurrent = hiddenCurrent.gather(2, backIndexVector)
                cellCurrent = cellCurrent.gather(2, backIndexVector)

                # Restore original shape for next rnn forward
                hiddenCurrent = hiddenCurrent.view(*original_state_size)
                cellCurrent = cellCurrent.view(*original_state_size)
                hiddenStates = (hiddenCurrent, cellCurrent)

            # Detecting endToken to end beams
            aliveVector = beamTokensTable[:, :, t:t + 1].ne(self.endToken)
            aliveBeams = aliveVector.data.long().sum()
            finalLen = t
            if aliveBeams == 0:
                break

        # Backtracking to get final beams
        beamTokensTable = beamTokensTable.data
        backIndices = backIndices.data

        # Keep this on when returning the top beam
        RECOVER_TOP_BEAM_ONLY = True

        tokenIdx = finalLen
        backID = backIndices[:, :, tokenIdx]
        tokens = []
        while (tokenIdx >= 0):
            tokens.append(beamTokensTable[:,:,tokenIdx].\
                        gather(1, backID).unsqueeze(2))
            backID = backIndices[:, :, tokenIdx].\
                        gather(1, backID)
            tokenIdx = tokenIdx - 1

        tokens.append(startTokenArray.unsqueeze(2).repeat(1, beamSize, 1).data)
        tokens.reverse()
        tokens = torch.cat(tokens, 2)
        seqLens = tokens.ne(self.endToken).long().sum(dim=2)

        if RECOVER_TOP_BEAM_ONLY:
            # 'tokens' has shape (batchSize, beamSize, maxLen)
            # 'seqLens' has shape (batchSize, beamSize)
            tokens = tokens[:, -1]  # Keep only top beam
            seqLens = seqLens[:, -1]

        return tokens, None