import copy
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from HCAM import HCAM

class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.terminals = []
        self.action_indexes = []
      
        self.curr_proc_adj = []
        self.batch_idxes = []
        self.norm_opes = []
        self.norm_macs = []
        self.mask_proc = []       


    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.terminals[:]
        del self.action_indexes[:]
        
        del self.curr_proc_adj[:]
        del self.batch_idxes[:]
        del self.norm_opes[:]
        del self.norm_macs[:]
        del self.mask_proc[:]


class PPO:
    def __init__(self, train_paras, model_paras, ppo_paras, num_envs=None):
        self.learning_rate = ppo_paras["learning_rate"]  # learning rate
        self.discount_factor = ppo_paras["discount_factor"]  # discount factor
        self.K_epoch = ppo_paras["K_epoch"]  # Update policy for K epochs
        self.betas = ppo_paras["betas"]
        self.clip_ratio = ppo_paras["clip_ratio"]  # clip ratio
        self.max_grad_norm = ppo_paras["max_grad_norm"]
        self.policy_coe = ppo_paras["policy_coe"]  # coefficient for policy loss
        self.value_coe = ppo_paras["value_coe"]  # coefficient for value loss
        self.entropy_coe = ppo_paras["entropy_coe"]  # coefficient for entropy term
        self.num_envs = num_envs  # Number of parallel instances
        self.minibatch_size = ppo_paras["minibatch_size"]  # batch size for updating
        self.device = train_paras["device"]  # PyTorch device

        self.policy = HCAM(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.learning_rate, betas=self.betas)
        self.MseLoss = nn.MSELoss()

    def update(self, memory, train_paras):
        device = train_paras["device"]
        old_curr_proc_adj = torch.stack(memory.curr_proc_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_norm_opes = torch.stack(memory.norm_opes, dim=0).transpose(0, 1).flatten(0, 1)
        old_norm_macs = torch.stack(memory.norm_macs, dim=0).transpose(0, 1).flatten(0, 1)
        old_mask_proc = torch.stack(memory.mask_proc, dim=0).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0, 1)
        memory_terminals = torch.stack(memory.terminals, dim=0).transpose(0, 1)
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0, 1).flatten(0, 1)
        old_action_indexes = torch.stack(memory.action_indexes, dim=0).transpose(0, 1).flatten(0, 1)
        
        rewards_envs = []
        initial_discounted_rewards_log = 0 

        for i in range(self.num_envs):
            rewards = []
            discounted_reward = 0
            
            for reward, is_terminal in zip(reversed(memory_rewards[i]), reversed(memory_terminals[i])):
                if is_terminal:
                    discounted_reward = 0  
                     
                discounted_reward = reward + (self.discount_factor * discounted_reward)
                rewards.insert(0, discounted_reward)
            
            initial_discounted_rewards_log += discounted_reward
            rewards_envs.append(torch.tensor(rewards, dtype=torch.float64).to(device))
            
        rewards_envs = torch.cat(rewards_envs)
        
        # --- Normalize Returns (Rewards) ---
        rewards_envs = (rewards_envs - rewards_envs.mean()) / (rewards_envs.std() + 1e-8)

        # --- PPO Optimization ---
        
        loss_epochs = 0
        full_batch_size = old_curr_proc_adj.size(0)
        indices = torch.arange(full_batch_size)
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epoch):
            # Shuffle indices for decorrelation
            shuffled_indices = indices[torch.randperm(full_batch_size)]
            
            for start_idx in range(0, full_batch_size, self.minibatch_size):
                end_idx = start_idx + self.minibatch_size
                batch_indices = shuffled_indices[start_idx:end_idx] # Select random batch indices
                
                # IMPORTANT: Use .detach() on old_logprobs for the ratio calculation
                old_logp_batch = old_logprobs[batch_indices].detach() 
                R_batch = rewards_envs[batch_indices]
                
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate(old_curr_proc_adj[batch_indices], 
                                         old_norm_opes[batch_indices],
                                         old_norm_macs[batch_indices],
                                         old_mask_proc[batch_indices],
                                         old_action_indexes[batch_indices])
                                         
                state_values = state_values.squeeze(-1)
                advantages = R_batch - state_values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Policy Ratio
                ratios = torch.exp(logprobs - old_logp_batch)

                # Clipped Surrogate Loss (Policy Loss)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = - self.policy_coe * torch.min(surr1, surr2)
                
                # Value Loss
                value_loss = self.value_coe * F.mse_loss(state_values, R_batch)
                
                # Entropy Loss
                entropy_loss = - self.entropy_coe * dist_entropy
                
                # Total Loss
                loss = policy_loss + value_loss + entropy_loss
                
                loss_epochs += loss.mean().detach()
                self.optimizer.zero_grad()
                loss.mean().backward()
                
                self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Return mean loss and mean discounted rewards for logging
        return loss_epochs.item() / (self.K_epoch * math.ceil(full_batch_size / self.minibatch_size)), \
               initial_discounted_rewards_log.item() / self.num_envs