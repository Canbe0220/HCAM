import copy
import torch
import math
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
from torch.distributions import Categorical
from mlp import MLPCritic, MLPActor

import torch
from torch import nn


import torch
import torch.nn as nn


class SEAM(nn.Module):
    def __init__(self, dim, num_heads=8, S=16, proj_drop=0.):
        super(SEAM, self).__init__()
        self.num_heads = num_heads
        self.S = S
        self.head_dim = dim // num_heads
        
        # Memory Units
        self.h1 = nn.Parameter(torch.empty(size=(self.head_dim, self.head_dim)))
        self.mk = nn.Parameter(torch.empty(size=(S, self.head_dim)))
        self.mv = nn.Parameter(torch.empty(size=(S, self.head_dim)))
        self.h2 = nn.Parameter(torch.empty(size=(self.head_dim, self.head_dim)))

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.h1)
        nn.init.xavier_normal_(self.mk)
        nn.init.xavier_normal_(self.mv)
        nn.init.xavier_normal_(self.h2)
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self, x):
        B, N, C = x.shape
        res = x
        x = x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        x = torch.matmul(x, self.h1)

        # Attention Map: (N, head_dim) @ (head_dim, S) -> (N, S)
        attn = torch.matmul(x, self.mk.transpose(0, 1))

        attn = attn.softmax(dim=-1)
        attn = attn / (attn.sum(dim=-2, keepdim=True) + 1e-8)

        # (N, S) @ (S, head_dim) -> (N, head_dim)
        x = torch.matmul(attn, self.mv)
        x = torch.matmul(x, self.h2)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x + res


class GRAM(nn.Module):
    def __init__(self, in_dim, out_dim, feat_drop=0., attn_drop=0.):
        super(GRAM, self).__init__()
        self.ope_dim = in_dim[0]
        self.mac_dim = in_dim[1]
        self.out_dim = out_dim
        self.num_heads = 2
        assert out_dim % self.num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = out_dim // self.num_heads
        self.nega_slope = 0.2

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.ope_w = nn.Linear(self.ope_dim, self.out_dim, bias=False)
        self.mac_w = nn.Linear(self.mac_dim, self.out_dim, bias=False)
        
        self.ope_alpha = nn.Parameter(torch.empty(size=(self.num_heads, self.head_dim, 1)))
        self.mac_alpha = nn.Parameter(torch.empty(size=(self.num_heads, self.head_dim, 1)))    
        
        self.leaky_relu = nn.LeakyReLU(self.nega_slope)

        if in_dim[1] != out_dim:
            self.res_fc = nn.Linear(in_dim[1], out_dim, bias=False)
        else:
            self.res_fc = None

        self.gate = nn.Linear(self.out_dim * 2, self.out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', self.nega_slope)
        nn.init.xavier_normal_(self.ope_w.weight, gain=gain)
        nn.init.xavier_normal_(self.mac_w.weight, gain=gain)
        nn.init.xavier_normal_(self.ope_alpha, gain=gain)
        nn.init.xavier_normal_(self.mac_alpha, gain=gain)
        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight)
        nn.init.xavier_normal_(self.gate.weight)
        
    def forward(self, curr_proc_batch, batch_idxes, feats):
        
        feat_ope = self.feat_drop(feats[0])
        feat_mac = self.feat_drop(feats[1])
            
        h_ope = self.ope_w(feat_ope) 
        h_mac = self.mac_w(feat_mac)

        B, N, _ = h_ope.shape
        M = h_mac.shape[1]
        h_ope_m = h_ope.view(B, N, self.num_heads, self.head_dim).transpose(1, 2) 
        h_mac_m = h_mac.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_ope = torch.matmul(h_ope_m, self.ope_alpha).squeeze(-1)
        attn_mac = torch.matmul(h_mac_m, self.mac_alpha).squeeze(-1)
        
        attn_ope = attn_ope.unsqueeze(-1) + attn_mac.unsqueeze(-2) 
        e_ijk = self.leaky_relu(attn_ope)

        mask_ijk = (curr_proc_batch[batch_idxes] == 1).unsqueeze(1)
        e_ijk = e_ijk.masked_fill(~mask_ijk, float('-9e10'))
        
        alpha_ijk = F.softmax(e_ijk, dim=-2)
        alpha_ijk = self.attn_drop(alpha_ijk)
        
        out_ope = torch.matmul(alpha_ijk.transpose(-1, -2), h_ope_m)
        
        out_ope = out_ope.transpose(1, 2).contiguous().view(B, M, self.out_dim)

        if self.res_fc is not None:
             out_res = self.res_fc(feat_mac)
        else:
             out_res = feat_mac

        combined = torch.cat([out_ope, out_res], dim=-1)
        g = torch.sigmoid(self.gate(combined))
        output = g * out_ope + (1 - g) * out_res

        return output


class HCAM(nn.Module):
    def __init__(self, model_paras):
        super(HCAM, self).__init__()

        self.device = model_paras["device"]
        self.in_ope_dim = model_paras["in_ope_dim"] 
        self.in_mac_dim = model_paras["in_mac_dim"]  
        self.out_ope_dim = model_paras["out_ope_dim"]
        self.out_mac_dim = model_paras["out_mac_dim"]
        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]

        self.actor_in_dim = model_paras["actor_in_dim"]
        self.critic_in_dim = model_paras["critic_in_dim"]
        self.actor_layer = self.critic_layer = model_paras["policy_layer"]
        self.actor_hidden_dim = self.critic_hidden_dim = model_paras["policy_hidden_dim"] 
        self.actor_out_dim = self.critic_out_dim = model_paras["policy_out_dim"] 
        
             
        self.get_opes = SEAM(self.in_ope_dim, self.num_heads)
        self.get_macs = GRAM((self.in_ope_dim, self.in_mac_dim), self.out_mac_dim, self.dropout, self.dropout)

        self.actor = MLPActor(self.actor_layer, self.actor_in_dim, self.actor_hidden_dim, self.actor_out_dim).to(self.device)
        self.critic = MLPCritic(self.critic_layer, self.critic_in_dim, self.critic_hidden_dim, self.critic_out_dim).to(self.device)


    def act_prob(self, state, memory, flag_train=True, flag_sample=True):

        '''
        probability distribution
        '''

        curr_proc_adj = state.curr_proc_batch
        batch_idxes = state.batch_idxes
        raw_opes = state.feat_opes_batch[batch_idxes]
        raw_macs = state.feat_macs_batch[batch_idxes]
        
        # Normalize
        mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)
        std_opes = torch.std(raw_opes, dim=-2, keepdim=True)
        norm_opes = (raw_opes - mean_opes) / (std_opes + 1e-8)

        mean_macs = torch.mean(raw_macs, dim=-2, keepdim=True)
        std_macs = torch.std(raw_macs, dim=-2, keepdim=True)
        norm_macs = (raw_macs - mean_macs) / (std_macs + 1e-8)

        h_opes = self.get_opes(norm_opes)
        h_macs = self.get_macs(curr_proc_adj[..., 0], batch_idxes, (norm_opes, norm_macs)) 
        h_pair = curr_proc_adj[batch_idxes]
     
        h_opes_pooled = h_opes.mean(dim=-2)
        h_macs_pooled = h_macs.mean(dim=-2)

        # expand and concatenate
        h_opes_expand = h_opes.unsqueeze(-2).expand(-1, -1, h_macs.size(-2), -1)
        h_macs_expand = h_macs.unsqueeze(-3).expand(-1, h_opes.size(-2), -1, -1)
        h_opes_pooled_expand = h_opes_pooled[:, None, None, :].expand_as(h_opes_expand)
        h_macs_pooled_expand = h_macs_pooled[:, None, None, :].expand_as(h_macs_expand)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch, state.end_ope_biases_batch, state.ope_step_batch)
        candidate_opes = ~(state.mask_job_procing_batch[batch_idxes] + state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_opes_expand[..., 0])
        idle_macs = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_opes_expand[..., 0])
        mask_proc = (curr_proc_adj[batch_idxes, ..., 0] == 1) & candidate_opes & idle_macs

        # actor MLP
        # h_actions = torch.cat((h_opes_expand, h_macs_expand, h_opes_pooled_expand, h_macs_pooled_expand, h_pair), dim=-1).transpose(1, 2)
        h_actions = torch.cat((h_opes_expand, h_macs_expand, h_pair), dim=-1).transpose(1, 2)
        mask = mask_proc.transpose(1, 2).flatten(1)
        
        #priority probability
        prob = self.actor(h_actions).flatten(1)
        prob[~mask] = float('-inf')
        action_probs = F.softmax(prob, dim=1)           

        if flag_sample:
            # using sample strategy during training
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        else:
            # using greedy strategy during validating and testing
            action_indexes = action_probs.argmax(dim=1)
        
        if flag_train == True:
            # Store memory data during training
            memory.logprobs.append(dist.log_prob(action_indexes))
            memory.action_indexes.append(action_indexes)
            memory.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memory.curr_proc_adj.append(copy.deepcopy(curr_proc_adj))
            memory.norm_opes.append(copy.deepcopy(norm_opes))
            memory.norm_macs.append(copy.deepcopy(norm_macs))
            memory.mask_proc.append(copy.deepcopy(mask_proc))
            
        # Calculate the machine, job and operation index based on the action index
        mas = (action_indexes / curr_proc_adj.size(1)).long()
        jobs = (action_indexes % curr_proc_adj.size(1)).long()
        opes = ope_step_batch[state.batch_idxes, jobs]         

        return torch.stack((opes, mas, jobs), dim=1).t()


    def evaluate(self, curr_proc_adj, norm_opes, norm_macs, mask_proc, action_indexes):
        batch_idxes = torch.arange(0, curr_proc_adj.size(0)).long()
        features = (norm_opes, norm_macs)

        h_opes = self.get_opes(norm_opes)
        h_macs = self.get_macs(curr_proc_adj[..., 0], batch_idxes, (norm_opes, norm_macs)) 
        h_pair = curr_proc_adj[batch_idxes]

        h_opes_pooled = h_opes.mean(dim=-2)
        h_macs_pooled = h_macs.mean(dim=-2)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
        h_opes_expand = h_opes.unsqueeze(-2).expand(-1, -1, h_macs.size(-2), -1)
        h_macs_expand = h_macs.unsqueeze(-3).expand(-1, h_opes.size(-2), -1, -1)
        h_opes_pooled_expand = h_opes_pooled[:, None, None, :].expand_as(h_opes_expand)
        h_macs_pooled_expand = h_macs_pooled[:, None, None, :].expand_as(h_macs_expand)

        # h_actions = torch.cat((h_opes_expand, h_macs_expand, h_opes_pooled_expand, h_macs_pooled_expand, h_pair), dim=-1).transpose(1, 2)
        h_actions = torch.cat((h_opes_expand, h_macs_expand, h_pair), dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_macs_pooled), dim=-1)
        prob = self.actor(h_actions).flatten(1)
        mask = mask_proc.transpose(1, 2).flatten(1)

        prob[~mask] = float('-inf')
        action_probs = F.softmax(prob, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_indexes)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze().double(), dist_entropys