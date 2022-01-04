import torch
import numpy as np
import copy
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
from tree import Tree, head_to_tree, tree_to_adj
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer


class SSDGCNClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        in_dim = opt.hidden_dim
        self.opt = opt
        self.gcn_model = GCNAbsaModel(embedding_matrix=embedding_matrix, opt=opt)
        self.classifier = nn.Linear(in_dim*2, opt.polarities_dim)  # 最后得到3分类

    def forward(self, inputs):
        outputs1, outputs2 = self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, outputs2), dim=-1)
        logits = self.classifier(final_outputs)
        penal = None

        return logits, penal


class GCNAbsaModel(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt
        self.embedding_matrix = embedding_matrix
        self.emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None  # POS emb
        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim,
                                     padding_idx=0) if opt.post_dim > 0 else None  # position emb
        embeddings = (self.emb, self.pos_emb, self.post_emb)
        self.in_dim = opt.embed_dim + opt.post_dim + opt.pos_dim
        # dual layer
        self.dual_encoder = DualEncoder(opt, embeddings)

    def forward(self, inputs):
        tok, asp, pos, head, dep, post, asp_mask, length, adj, dist = inputs  # unpack inputs
        maxlen = max(length.data)
        asp_mask = asp_mask[:, :maxlen]  # torch.Size([bs, maxlen])
        if self.opt.parseadj:
            adj_dep = adj[:, :maxlen, :maxlen].float()
        else:
            def inputs_to_tree_reps(head, words, l):
                trees = [head_to_tree(head[i], words[i], l[i]) for i in range(len(l))]
                adj = [tree_to_adj(
                    maxlen, tree, directed=self.opt.direct, self_loop=self.opt.loop).reshape(1, maxlen, maxlen) for tree in trees]
                adj = np.concatenate(adj, axis=0)
                adj = torch.from_numpy(adj)
                return adj.cuda()

            adj_dep = inputs_to_tree_reps(head.data, tok.data, length.data)  # torch.Size([16, maxlen, maxlen])

        h_syn, h_sem = self.dual_encoder(adj_dep, inputs)

        asp_wn = asp_mask.sum(dim=1).unsqueeze(-1)  # aspect words num
        mask = asp_mask.unsqueeze(-1).repeat(1, 1, self.opt.hidden_dim)  # mask for h
        outputs1 = (h_syn * mask).sum(dim=1) / asp_wn
        outputs2 = (h_sem * mask).sum(dim=1) / asp_wn
        return outputs1, outputs2


class DualEncoder(nn.Module):
    def __init__(self, opt, embeddings):
        super(DualEncoder, self).__init__()
        self.opt = opt
        self.num_layers = opt.num_layers
        self.in_dim = opt.embed_dim + opt.post_dim + opt.pos_dim
        self.emb, self.pos_emb, self.post_emb = embeddings

        # rnn layer = Bi-LSTM
        self.rnn = nn.LSTM(self.in_dim, opt.rnn_hidden, opt.rnn_layers, batch_first=True,
                           dropout=opt.rnn_dropout, bidirectional=opt.bidirect)
        if opt.bidirect:
            self.in_dim = opt.rnn_hidden * 2
        else:
            self.in_dim = opt.rnn_hidden

        # drop out
        self.in_drop = nn.Dropout(opt.input_dropout)
        self.rnn_drop = nn.Dropout(opt.rnn_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # gcn layer
        self.gcn_syn = nn.ModuleList()
        self.gcn_sem = nn.ModuleList()
        self.mhsa_sem = nn.ModuleList()
        self.ffn_sem = nn.ModuleList()

        # mask gcn
        self.linear_double_syn = nn.Linear(opt.rnn_hidden * 4, opt.rnn_hidden * 2)
        self.linear_double_sem = nn.Linear(opt.rnn_hidden * 4, opt.rnn_hidden * 2)

        # GCN层, syn GCN + sem MHSA
        # ########## MHGCN
        if opt.mhgcn:
            self.gcn_syn.append(MultiHeadGCN(opt, opt.gcn_heads, opt.rnn_hidden * 2, opt.hidden_dim * 2))
        else:
            self.gcn_syn.append(GCN(opt, opt.rnn_hidden * 2, opt.hidden_dim*2))  # 消融

        # 语义 MHSA + PCT
        self.mhsa_sem.append(MultiHeadAttention(opt, opt.gcn_heads, opt.rnn_hidden * 2))
        self.ffn_sem.append(PointwiseConv(opt.rnn_hidden * 2, dropout=opt.input_dropout))
        # self.ffn_sem.append(nn.Linear(opt.rnn_hidden*2, opt.hidden_dim))
        for i in range(1, self.opt.num_layers):
            # ########## MHGCN
            if opt.mhgcn:
                self.gcn_syn.append(MultiHeadGCN(opt, opt.gcn_heads, opt.hidden_dim * 2, opt.hidden_dim * 2))
            else:
                self.gcn_syn.append(GCN(opt, opt.rnn_hidden * 2, opt.hidden_dim*2))  # TODO 消融 MHGCN --> GCN
            self.mhsa_sem.append(MultiHeadAttention(opt, opt.gcn_heads, opt.rnn_hidden * 2))
            self.ffn_sem.append(PointwiseConv(opt.rnn_hidden * 2, dropout=opt.input_dropout))

        # learnable hyperparameter，最后的 α
        self.alpha = nn.Parameter(torch.tensor([0.5]))

        # #### #fully connect Layer
        # self.linear = nn.Linear(4 * opt.hidden_dim, opt.hidden_dim)  # TODO (816,204) all
        self.linear = nn.Linear(2 * opt.hidden_dim, opt.hidden_dim)  # (408,204)
        self.final_drop = nn.Dropout(opt.input_dropout)

        self.hid_linear = nn.Linear(opt.hidden_dim * 2, opt.hidden_dim)  # 408-->204,再resnet

    # 拼接三种词嵌入
    def create_embs(self, tok, pos, post):
        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.opt.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.opt.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)  # input_dropout
        return embs

    # Bi-LSTM，input：embs, length, embs.size(0)
    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(self.opt, batch_size, 1, True)  # 1=一层RNN
        rnn_inputs = pack_padded_sequence(rnn_inputs, seq_lens.cpu(), batch_first=True,
                                          enforce_sorted=False)  # 输入压缩为可变长序列。先打包再填充
        rnn_outputs, (_, _) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def create_adj_mask(self, rnn_hidden):
        score_mask = torch.matmul(rnn_hidden, rnn_hidden.transpose(-2, -1))  # 句子自己和自己相乘，没有词的地方为空
        score_mask = (score_mask == 0)  # =0的值为True
        return score_mask

    def Dense(self, inputs, seq_lens):
        # padd
        inputs = self.dense(inputs)
        inputs_unpad = pack_padded_sequence(inputs, seq_lens.cpu(), batch_first=True)
        outputs, _ = pad_packed_sequence(inputs_unpad, batch_first=True)
        return outputs

    # mask dep_dist
    def feature_dynamic_mask(self, text, asp, asp_mask=None, distances_input=None):
        texts = text.cpu()  # batch_size x seq_len x rnn*2
        asps = asp.cpu()  # batch_size x aspect_len
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
            mask_len = self.opt.syn_srd
        if asp_mask is not None:
            asp_mask = asp_mask.cpu()
            mask_len = self.opt.sem_srd
        masked_text_vec = np.ones((text.size(0), text.size(1), self.opt.rnn_hidden * 2),
                                  dtype=np.float32)  # batch_size x seq_len x rnn hidden size*2
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):  # For each sample
            if distances_input is None:
                # asp_len = np.count_nonzero(asps[asp_i])  # Calculate aspect length
                asp_len = torch.count_nonzero(asps[asp_i])
                try:
                    # asp_begin = np.argwhere(texts[text_i] == asps[asp_i][0])[0][0]
                    asp_begin = torch.nonzero(asp_mask[asp_i])[0]
                except:
                    continue
                # Mask begin -> Relative position of an aspect vs the mask
                if asp_begin >= mask_len:
                    mask_begin = asp_begin - mask_len
                else:
                    mask_begin = 0
                for i in range(mask_begin):  # Masking to the left
                    masked_text_vec[text_i][i] = np.zeros(self.opt.rnn_hidden * 2, dtype=np.float)
                for j in range(asp_begin + asp_len + mask_len, text.size(1)):  # Masking to the right
                    masked_text_vec[text_i][j] = np.zeros(self.opt.rnn_hidden * 2, dtype=np.float)
            else:
                distances_i = distances_input[text_i][:len(texts[1])]  # 按行取
                for i, dist in enumerate(distances_i):
                    if dist > mask_len:
                        masked_text_vec[text_i][i] = np.zeros(self.opt.rnn_hidden * 2, dtype=np.float)

        masked_text_vec = torch.from_numpy(masked_text_vec)
        return masked_text_vec.to(self.opt.device)

    # weighted
    def feature_dynamic_weighted(self, hid, asp, asp_mask=None, distances_input=None):
        texts = hid.cpu()
        asps = asp.cpu().numpy()
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
            mask_len = self.opt.syn_srd
        if asp_mask is not None:
            asp_mask = asp_mask.cpu().numpy()
            mask_len = self.opt.sem_srd
        masked_text_raw_indices = np.ones((hid.size(0), hid.size(1), self.opt.rnn_hidden * 2),
                                          dtype=np.float32)  # batch x seq x dim
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            if distances_input is None:  # 语义
                asp_len = np.count_nonzero(asps[asp_i])
                try:
                    asp_begin = np.nonzero(asp_mask[asp_i])[0][0]
                    asp_avg_index = (asp_begin * 2 + asp_len) / 2  # central position
                except:
                    continue
                distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)  # , dtype=np.float32
                for i in range(1,
                               np.count_nonzero(texts[text_i]) - 1):  # 1-35 从1开始，0为CLS，mask_len=3，不mask算上自己的前两个，后两个
                    srd = abs(i - asp_avg_index) + asp_len / 2
                    if srd > mask_len:
                        distances[i] = 1 - (srd - mask_len) / np.count_nonzero(texts[text_i])
                    else:
                        distances[i] = 1
                for i in range(len(distances)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
            else:
                distances_i = distances_input[text_i]  # distances of batch i-th
                for i, dist in enumerate(distances_i):
                    if dist > mask_len:
                        distances_i[i] = 1 - (dist - mask_len) / np.count_nonzero(texts[text_i])
                    else:
                        distances_i[i] = 1

                for i in range(len(distances_i)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances_i[i]

        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    # ############## Model Start ################
    def forward(self, adj, inputs):
        tok, asp, pos, head, dep, post, asp_mask, length, _, dist = inputs  # unpack inputs
        src_mask = (tok != 0).unsqueeze(-2)  # torch.Size([16, 1, 85])
        maxlen = max(length.data)
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]  # torch.Size([16, 64, 1])

        # 三重embedding
        embs = self.create_embs(tok, pos, post)  # torch.Size([16, 85, 360])
        # Bi-LSTM encoding
        self.rnn.flatten_parameters()
        hidden = self.rnn_drop(
            self.encode_with_rnn(embs, length, embs.size(0)))  # [batch_size, seq_len, rnn_hidden*2] torch.Size([16, 64, 100])
        # bi-lism 句子级特征提取
        score_mask = self.create_adj_mask(hidden)  # score_mask=0的值为True torch.Size([32, 41, 41])  torch.Size([16, 64, 64])

        h_syn = []  # 三层GCN的输出
        h_sem = []
        h_syn_res = []
        h_sem_res = []
        h_syn_mask = []
        h_sem_mask = []

        # Syn MHGCN #####
        h_syn.append(self.gcn_syn[0](adj, hidden, score_mask,
                                     first_layer=True))  # 句法图和BiLSTM输出，一起输入三层SynGCN , first_layer=True torch.Size([16, 64, 100])
        # Sem MHSA ######
        h_sem.append(self.mhsa_sem[0](hidden, hidden, score_mask, first_layer=True))
        h_sem[0] = (self.ffn_sem[0](h_sem[0], first_layer=True))
        # resnet
        if self.opt.shortcut:
            h_syn_res.append(hidden + h_syn[0])
            h_sem_res.append(hidden + h_sem[0])  # 32,41,408
        else:
            h_syn_res.append(h_syn[0])
            h_sem_res.append(h_sem[0])
        # mask h_sem_res = local + global
        if self.opt.local_sem_focus == 'sem_cdm':
            masked_sem_vec = self.feature_dynamic_mask(h_sem_res[0], asp, asp_mask)
            h_sem_mask.append(torch.mul(h_sem_res[0], masked_sem_vec))
            # local_h_sem = torch.mul(h_sem_res[0], masked_sem_vec)
            # h_sem_res[0] = torch.cat((h_sem_res[0], local_h_sem), dim=-1)
            # # h_sem_res[0] = self.sem_mean_pool(h_sem_res[0])
            # h_sem_res[0] = self.linear_double_sem(h_sem_res[0])

        if self.opt.local_syn_focus == 'syn_cdm':
            masked_syn_vec = self.feature_dynamic_mask(h_syn_res[0], asp, distances_input=dist)
            h_syn_mask.append(torch.mul(h_syn_res[0], masked_syn_vec))
            # local_h_syn = torch.mul(h_syn_res[0], masked_syn_vec)
            # h_syn_res[0] = torch.cat((h_syn_res[0], local_h_syn), dim=-1)  # 方案2
            # # h_syn[0] = self.syn_mean_pool(h_syn_global_local)
            # h_syn_res[0] = self.linear_double_syn(h_syn_res[0])

        # weight h_syn_res = local + global
        if self.opt.local_sem_focus == 'sem_cdw':
            masked_sem_vec = self.feature_dynamic_weighted(tok, asp, asp_mask)
            local_h_sem = torch.mul(h_sem_res[0], masked_sem_vec)
            h_sem_res[0] = torch.cat((h_sem_res[0], local_h_sem), dim=-1)
            # h_sem_res[0] = self.sem_mean_pool(h_sem_res[0])
            h_sem_res[0] = self.linear_double_sem(h_sem_res[0])

        # weight h_syn_res = local + global
        if self.opt.local_syn_focus == 'syn_cdw':
            masked_syn_vec = self.feature_dynamic_weighted(h_syn_res[0], asp, asp_mask)
            local_h_syn = torch.mul(h_syn_res[0], masked_syn_vec)
            h_syn_res[0] = torch.cat((h_syn_res[0], local_h_syn), dim=-1)
            h_syn_res[0] = self.linear_double_syn(h_syn_res[0])

        # ########## 多层
        for i in range(self.opt.num_layers - 1):
            # ######## MHSA、MultiGCN
            h_syn.append(self.gcn_syn[i + 1](adj, h_syn_res[i], score_mask))  # h_syn[1]
            h_sem.append(self.mhsa_sem[i + 1](h_sem_res[i], h_sem_res[i], score_mask))
            h_sem[i + 1] = self.ffn_sem[i + 1](h_sem[i + 1])
            if self.opt.shortcut:
                h_syn_res.append(hidden + h_syn_res[i] + h_syn[i + 1] + h_syn_mask[i])
                h_sem_res.append(hidden + h_sem_res[i] + h_sem[i + 1] + h_sem_mask[i])
            else:
                h_syn_res.append(h_syn[i + 1] + h_syn_mask[i])
                h_sem_res.append(h_sem[i + 1] + h_sem_mask[i])
        # h_final = torch.cat((h_syn_final, h_sem_final), dim=-1)
        h_syn_pool = torch.div(torch.sum(h_syn_res[-1], dim=1), length.view(length.size(0), 1))
        h_sem_pool = torch.div(torch.sum(h_sem_res[-1], dim=1), length.view(length.size(0), 1))
        # 32,408

        # ############ 双通道拼接
        # out = torch.cat((h_syn_pool, h_sem_pool), dim=-1)  # TODO 直接拼接32,816
        out = self.alpha * h_syn_pool + (1 - self.alpha) * h_sem_pool  # 32,408
        # linear
        # outputs = torch.tanh(self.linear(h_syn_res[0]))  # MHGCN
        # ############
        outputs = torch.tanh(self.linear(out))  # all 32,204
        outputs = self.final_drop(outputs)
        return h_syn_res[-1], h_sem_res[-1]  # 32,51,408
        # return outputs, h_syn_res[-1], h_sem_res[-1]  # 32,51,408


# 构造h0，c0的0矩阵
# def rnn_zero_state(opt, batch_size, num_layers, bidirectional=True):
#     total_layers = num_layers * 2 if bidirectional else num_layers
#     state_shape = (total_layers, batch_size, opt.rnn_hidden)
#     h0 = c0 = torch.zeros(*state_shape)
#     return h0.to(opt.device), c0.to(opt.device)

def rnn_zero_state(opt, batch_size, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, opt.rnn_hidden)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


class GCN(nn.Module):
    def __init__(self, opt, input_dim, output_dim):
        super(GCN, self).__init__()
        self.opt = opt
        self.drop = nn.Dropout(opt.gcn_dropout)

        # gcn layer
        self.W = nn.Linear(input_dim, output_dim, bias=False)  # 是否加bias

    def forward(self, adj_dep, inputs, score_mask, first_layer=True):
        denom = adj_dep.sum(2).unsqueeze(2) + 1  # norm adj 度矩阵：表示连接节点数
        Ax = adj_dep.bmm(inputs)
        AxW = self.W(Ax)
        AxW = AxW / denom
        gAxW = F.relu(AxW)
        out = gAxW if first_layer else self.drop(gAxW)
        return out  # （16,41,204)


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z, mask):
        w = self.project(z)
        w = w.masked_fill(mask[:, :, 0:1], -1e10)
        w = torch.softmax(w, dim=1)
        return w * z, w


# 语义提取通道 1、MHSA
class MultiHeadAttention(nn.Module):
    # d_model:hidden_dim，h:gcn_heads
    def __init__(self, opt, gcn_heads, hidden_dim):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % gcn_heads == 0

        self.d_k = int(hidden_dim // gcn_heads)
        self.gcn_heads = gcn_heads
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(hidden_dim, hidden_dim)) for _ in range(2)])
        self.dropout = nn.Dropout(opt.mhsa_dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def attention(self, query, key, score_mask, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if score_mask is not None:
            scores = scores.masked_fill(score_mask.unsqueeze(1), -1e9)  # 在mask为True时，用-1e9填充张量元素。
        b = ~(score_mask.unsqueeze(1)[:, :, :, 0:1])
        p_attn = F.softmax(scores, dim=-1) * b.float()
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn

    def forward(self, query, key, score_mask, first_layer=False):
        residual = query
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.gcn_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = self.attention(query, key, score_mask, dropout=self.dropout)  # (32,3,41,41)
        output = torch.matmul(attn, query)
        output = torch.cat(torch.split(output, 1, dim=1), dim=-1).squeeze(1)
        output = self.proj(output)
        output = output if first_layer else self.dropout(output)
        output = self.layer_norm(residual + output)
        return output


# 语义提取通道 2、逐点卷积
class PointwiseConv(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_hid, d_inner_hid=None, d_out=None, dropout=0):
        super(PointwiseConv, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        if d_out is None:
            d_out = d_inner_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_out, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_hid)

    def forward(self, x, first_layer=False):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        output = self.layer_norm(x + output)
        return output


# 句法提取通道
class MultiHeadGCN(nn.Module):
    def __init__(self, opt, gcn_heads, input_dim, output_dim):
        super(MultiHeadGCN, self).__init__()
        self.opt = opt
        self.d_k = int(input_dim // gcn_heads)
        self.h_k = int(output_dim // gcn_heads)
        self.gcn_heads = gcn_heads
        self.mapped = nn.Linear(input_dim, output_dim)
        # gcn layer
        self.W = nn.Linear(self.h_k, self.h_k, bias=False)  # 是否加bias
        self.drop = nn.Dropout(opt.gcn_dropout)

    def forward(self, adj_dep, hidden, score_mask, first_layer=False):
        bs = hidden.size(0)
        seq_len = hidden.size(1)
        score_mask = score_mask.unsqueeze(1).repeat(1, self.gcn_heads, 1, 1)
        sem_hidden = self.mapped(hidden)
        sem_hidden = sem_hidden.view(bs, seq_len, self.gcn_heads, -1)  # torch.Size([16, 64, 4, 25])
        sem_hidden = sem_hidden.permute(2, 0, 1, 3)  # (4,32,41,102) torch.Size([4, 16, 64, 25])
        denom = adj_dep.sum(2).unsqueeze(2) + 1  # (32,41,1)
        Ax = adj_dep.matmul(sem_hidden)  # (4,32,41,102)
        AxW = self.W(Ax)
        AxW = AxW / denom
        # out = self.drop(F.relu(AxW))
        out = F.relu(AxW)
        out = out if first_layer else self.drop(out)
        out = torch.cat(torch.split(out, 1, dim=0), dim=-1).squeeze(0)  # 多头求平均也行
        return out
