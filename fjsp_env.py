import sys
import gym
import torch

from dataclasses import dataclass
from data_utils import load_fjs, nums_detec, read_json, write_json
import numpy as np
import random
import copy

from gym.envs.registration import register

# Registrar for the gym environment
# https://www.gymlibrary.ml/content/environment_creation/ for reference
register(
    id='fjsp-v0',  # Environment name (including version number)
    entry_point='fjsp_env:FJSPEnv',  # The location of the environment class, like 'foldername.filename:classname'
)

@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # static
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None

    # dynamic
    batch_idxes: torch.Tensor = None
    curr_proc_batch: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_macs_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch:  torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None

    def update(self, batch_idxes, curr_proc_batch, feat_opes_batch, feat_macs_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time):
        self.batch_idxes = batch_idxes
        self.curr_proc_batch = curr_proc_batch
        self.feat_opes_batch = feat_opes_batch
        self.feat_macs_batch = feat_macs_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time

class FJSPEnv(gym.Env):
    '''
    FJSP environment
    '''
    def __init__(self, case, env_paras, data_source='case'):
        '''
        :param case: The instance generator or the addresses of the instances
        :param env_paras: A dictionary of parameters for the environment
        :param data_source: Indicates that the instances came from a generator or files
        '''

        # load paras
        # static
        self.batch_size = env_paras["batch_size"]  # Number of parallel instances during training
        self.num_jobs = env_paras["num_jobs"]  # Number of jobs
        self.num_mas = env_paras["num_mas"]  # Number of machines
        self.paras = env_paras  # Parameters
        self.device = env_paras["device"]  # Computing device for PyTorch
        # load instance
        num_data = 4  # The amount of data extracted from instance
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        lines = []
        if data_source=='case':  # Generate instances through generators
            for i in range(self.batch_size):
                lines.append(case.get_case(i)[0])  # Generate an instance and save it
                num_jobs, num_mas, num_opes = nums_detec(lines[i])
                # Records the maximum number of operations in the parallel instances
                self.num_opes = max(self.num_opes, num_opes)
        else:  # Load instances from files
            for i in range(self.batch_size):
                with open(case[i]) as file_object:
                    line = file_object.readlines()
                    lines.append(line)
                num_jobs, num_mas, num_opes = nums_detec(lines[i])
                self.num_opes = max(self.num_opes, num_opes)
        # load feats
        for i in range(self.batch_size):
            load_data = load_fjs(lines[i], num_mas, self.num_opes)
            for j in range(num_data):
                tensors[j].append(load_data[j])

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()

        # static feats
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[2], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[3], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.time = torch.zeros(self.batch_size)  # Current time of the environment
        self.N = torch.zeros(self.batch_size).int()  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

        '''
        features, dynamic
            ope:
                Number of neighboring machines
                Average processing time
                Estimated earliest start time
                Processing status of the job
                Completion status of the job 
                Estimated completion time of the job
                Sum of the average processing time of unscheduled operations remaining in the job
                Number of unscheduled operations remaining in the job

            mac:
                Number of all neighboring operations
                Average processing time of all neighboring operations
                Processing status
                Earliest idle time
                
            pair:
                Current adjacency matrix
                Current processing time
                Ratio of processing time to the maximum processing time of machines
                Ratio of processing time to the maximum processing time of operations
        '''

        curr_proc_batch = torch.zeros(size=(self.batch_size, num_jobs, num_mas, self.paras["pair_dim"]))
        feat_opes_batch = torch.zeros(size=(self.batch_size, num_jobs, self.paras["ope_dim"]))
        feat_macs_batch = torch.zeros(size=(self.batch_size, num_mas, self.paras["mac_dim"]))
        
        batch_indices = torch.arange(self.batch_size).unsqueeze(1).expand(-1, self.num_jobs)
        job_indices = torch.arange(self.num_jobs).unsqueeze(0).expand(self.batch_size, -1)

        start_opes = self.ope_step_batch[batch_indices, job_indices]
        end_opes = self.end_ope_biases_batch[batch_indices, job_indices]

        self.max_proc = torch.max(self.proc_times_batch)
        self.min_proc = torch.min(self.proc_times_batch)
        curr_proc_batch[..., 0] = self.ope_ma_adj_batch[batch_indices, start_opes]
        curr_proc_batch[..., 1] = (self.proc_times_batch[batch_indices, start_opes] - self.min_proc).div(self.max_proc - self.min_proc + 1e-8)
        curr_proc_batch[..., 2] = curr_proc_batch[..., 1].div(torch.max(curr_proc_batch[..., 1], dim=-1, keepdim=True)[0] + 1e-8)
        curr_proc_batch[..., 3] = curr_proc_batch[..., 1].div(torch.max(curr_proc_batch[..., 1], dim=-2, keepdim=True)[0] + 1e-8)
        
        mean_proc_time = torch.sum(self.proc_times_batch, dim=-1).div(torch.count_nonzero(self.ope_ma_adj_batch, dim=-1) + 1e-8)   
        cum_time = torch.cumsum(mean_proc_time, dim=-1)

        feat_opes_batch[..., 0] = torch.count_nonzero(self.ope_ma_adj_batch[batch_indices, start_opes], dim=-1)
        feat_opes_batch[..., 1] = mean_proc_time[batch_indices, start_opes]

        feat_opes_batch[..., 5] = cum_time[batch_indices, end_opes] - cum_time[batch_indices, start_opes] \
                                + mean_proc_time[batch_indices, start_opes]
        feat_opes_batch[..., 6] = feat_opes_batch[..., 5]
        feat_opes_batch[..., 7] = self.nums_ope_batch                
                   
        feat_macs_batch[..., 0] = torch.count_nonzero(self.ope_ma_adj_batch, dim=-2)
        feat_macs_batch[..., 1] = torch.sum(self.proc_times_batch, dim=-2).div(feat_macs_batch[..., 0] + 1e-8)

        self.curr_proc_batch = curr_proc_batch
        self.feat_opes_batch = feat_opes_batch
        self.feat_macs_batch = feat_macs_batch
        
        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, num_mas), dtype=torch.bool, fill_value=False)

        
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, :, 5], dim=1)[0]  # shape: (batch_size)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)

        self.state = EnvState(batch_idxes=self.batch_idxes, curr_proc_batch=self.curr_proc_batch,
                              feat_opes_batch=self.feat_opes_batch, feat_macs_batch=self.feat_macs_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes)

        # Save initial data for reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_curr_proc_batch = copy.deepcopy(self.curr_proc_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_macs_batch = copy.deepcopy(self.feat_macs_batch)
        self.old_state = copy.deepcopy(self.state)

    def step(self, actions):
        '''
        Environment transition function
        '''
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        self.N += 1

        # Removed unselected O-M arcs of the scheduled operations
        ope_ma_adj = self.ope_ma_adj_batch[self.batch_idxes, opes, :]
        proc_time_adj = self.proc_times_batch[self.batch_idxes, opes, :]
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        # Update other variable according to actions
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True
        self.mask_ma_procing_batch[self.batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch==self.end_ope_biases_batch+1,
                                                 True, self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        # min_proc_batch = min_proc_time(self.proc_times_batch, -1)

        mean_proc_time = torch.sum(self.proc_times_batch, dim=-1).div(torch.count_nonzero(self.ope_ma_adj_batch, dim=-1) + 1e-8)
        end_opes = self.end_ope_biases_batch[self.batch_idxes, jobs]
        next_opes = torch.where(opes==end_opes, end_opes, opes + 1)
        
        self.curr_proc_batch[self.batch_idxes, jobs, :, 0] = self.ope_ma_adj_batch[self.batch_idxes, next_opes, :].float()
        self.curr_proc_batch[self.batch_idxes, jobs, :, 1] = (self.proc_times_batch[self.batch_idxes, next_opes, :]  - self.min_proc) / (self.max_proc - self.min_proc + 1e-8)
        self.curr_proc_batch[..., 2] = self.curr_proc_batch[..., 1].div(torch.max(self.curr_proc_batch[..., 1], dim=-1, keepdim=True)[0] + 1e-8)
        self.curr_proc_batch[..., 3] = self.curr_proc_batch[..., 1].div(torch.max(self.curr_proc_batch[..., 1], dim=-2, keepdim=True)[0] + 1e-8)


        self.feat_opes_batch[self.batch_idxes, jobs, 0] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, next_opes, :], dim=-1).float()
        self.feat_opes_batch[self.batch_idxes, jobs, 1] = mean_proc_time[self.batch_idxes, next_opes]
        self.feat_opes_batch[self.batch_idxes, jobs, 2] = self.time[self.batch_idxes] + self.proc_times_batch[self.batch_idxes, opes, mas] 

        self.feat_opes_batch[self.batch_idxes, :, 3] = self.mask_job_procing_batch[self.batch_idxes, :].float()
        self.feat_opes_batch[self.batch_idxes, :, 4] = self.mask_job_finish_batch[self.batch_idxes, :].float()
        
        cum_time = torch.cumsum(mean_proc_time, dim=-1)
        total_sum = cum_time[self.batch_idxes, end_opes] - cum_time[self.batch_idxes, opes]
        self.feat_opes_batch[self.batch_idxes, jobs, 5] = self.feat_opes_batch[self.batch_idxes, jobs, 2] + total_sum
        self.feat_opes_batch[self.batch_idxes, jobs, 6] = total_sum
        self.feat_opes_batch[self.batch_idxes, jobs, 7] -= 1                            


        self.feat_macs_batch[self.batch_idxes, :, 0] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, ...], dim=-2).float()
        self.feat_macs_batch[self.batch_idxes, :, 1] = torch.sum(self.proc_times_batch[self.batch_idxes, ...], dim=-2) \
                                                       .div(self.feat_macs_batch[self.batch_idxes, :, 0] + 1e-8)
        self.feat_macs_batch[self.batch_idxes, :, 2] = self.mask_ma_procing_batch[self.batch_idxes, :].float()
        self.feat_macs_batch[self.batch_idxes, mas, 3] = self.time[self.batch_idxes] + self.proc_times_batch[self.batch_idxes, opes, mas]
        

        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))
        self.machines_batch[self.batch_idxes, mas, 1] = self.time[self.batch_idxes] + self.proc_times_batch[self.batch_idxes, opes, mas]
        self.machines_batch[self.batch_idxes, mas, 2] += self.proc_times_batch[self.batch_idxes, opes, mas]
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()

        max = torch.max(self.feat_opes_batch[:, :, 5], dim=1)[0]
        self.reward_batch = self.makespan_batch - max
        self.makespan_batch = max

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        flag_trans_2_next_time = self.if_no_eligible()
        while ~((~((flag_trans_2_next_time==0) & (~self.done_batch))).all()):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()

        # Update the vector for uncompleted instances
        mask_finish = (self.N+1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]

        # Update state of the environment
        self.state.update(self.batch_idxes, self.curr_proc_batch, self.feat_opes_batch, self.feat_macs_batch, self.proc_times_batch,
            self.ope_ma_adj_batch, self.mask_job_procing_batch, self.mask_job_finish_batch, self.mask_ma_procing_batch,
                          self.ope_step_batch, self.time)
        return self.state, self.reward_batch, self.done_batch

    def if_no_eligible(self):
        '''
        Check if there are still O-M pairs to be processed
        '''
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
                                     self.end_ope_biases_batch, self.ope_step_batch)
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                        self.proc_times_batch.size(2)))
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
            op_proc_time)
        flag_trans_2_next_time = torch.sum(torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
                                           dim=[1, 2])
        
        # shape: (batch_size)
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        '''
        Transit to the next time
        '''
        # need to transit
        flag_need_trans = (flag_trans_2_next_time==0) & (~self.done_batch)
        # available_time of machines
        a = self.machines_batch[:, :, 1]
        # remain available_time greater than current time
        b = torch.where(a > self.time[:, None], a, torch.max(self.feat_opes_batch[:, :, 5]) + 1.0)
        # Return the minimum value of available_time (the time to transit to)
        c = torch.min(b, dim=1)[0]
        # Detect the machines that completed (at above time)
        d = torch.where((a == c[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)
        # The time for each batch to transit to or stay in
        e = torch.where(flag_need_trans, c, self.time)
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1
        self.machines_batch = aa.transpose(1, 2)

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        opes_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[opes_index[0], opes_index[1]].long()
        batch_idxes = opes_index[0]

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

        self.feat_opes_batch[self.batch_idxes, :, 3] = self.mask_job_procing_batch[self.batch_idxes, :].float() 
        self.feat_opes_batch[self.batch_idxes, :, 4] = self.mask_job_finish_batch[self.batch_idxes, :].float()
        self.feat_macs_batch[self.batch_idxes, :, 2] = self.mask_ma_procing_batch[self.batch_idxes, :].float()


    def reset(self):
        '''
        Reset the environment to its initial state
        '''
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.curr_proc_batch = copy.deepcopy(self.old_curr_proc_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_macs_batch = copy.deepcopy(self.old_feat_macs_batch)
        self.state = copy.deepcopy(self.old_state)

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)

        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, :, 5], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        return self.state
