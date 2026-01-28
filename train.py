import copy
import json
import os
import random
import time
from collections import deque

import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

from data_utils import CaseGenerator
import ppo
import fjsp_env

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_validate_env(valid_paras):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    file_path = "./data_dev/{0}{1}/".format(valid_paras["num_jobs"], str.zfill(str(valid_paras["num_mas"]),2))
    valid_data_files = os.listdir(file_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path+valid_data_files[i]
    env = gym.make('fjsp-v0', case=valid_data_files, env_paras=valid_paras, data_source='file')
    return env

def validate(valid_paras, env, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = valid_paras["batch_size"]
    memory = ppo.Memory()
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = model_policy.act_prob(state, memory, flag_train=False, flag_sample=False)
        state, rewards, dones = env.step(actions)
        done = dones.all()

    makespan = copy.deepcopy(env.makespan_batch.mean())
    env.reset()
    print('validating time: ', time.time() - start, '\n')
    return makespan

def main():
    setup_seed(100)
    # PyTorch initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)

    # Load config and init objects
    with open("./param.json", 'r') as load_f:
        load_dict = json.load(load_f)
    train_paras = load_dict["train_paras"]
    model_paras = load_dict["model_paras"]
    ppo_paras = load_dict["ppo_paras"]

    train_paras["device"] = device
    model_paras["device"] = device
    valid_paras = copy.deepcopy(train_paras)
    valid_paras["batch_size"] = train_paras["valid_size"]

    num_jobs = train_paras["num_jobs"]
    num_mas = train_paras["num_mas"]

    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)

    memory = ppo.Memory()
    model = ppo.PPO(train_paras, model_paras, ppo_paras, num_envs=train_paras["batch_size"])
    env_valid = get_validate_env(valid_paras)  # Create an environment for validation
    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float('inf')
 
    vis = Visdom(env=train_paras["visdom_name"])

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './train_results/train_{0}'.format(str_time)
    os.makedirs(save_path)

    valid_results = []
    train_rewards = []

    # Training curve storage path (average makespan of validation set)
    writer_makesapn = pd.ExcelWriter('{0}/training_makespan_{1}.xlsx'.format(save_path, str_time))
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data_file.to_excel(writer_makesapn, sheet_name='Sheet1', index=False)
    writer_makesapn._save()

    # Training curve of rewards
    writer_reward = pd.ExcelWriter('{0}/training_reward_{1}.xlsx'.format(save_path, str_time))
    data_file = pd.DataFrame(np.arange(1, 1001, 1), columns=["iterations"])
    data_file.to_excel(writer_reward, sheet_name='Sheet1', index=False)
    writer_reward._save()

    # Start training iteration
    start_time = time.time()
    env = None
    for i in range(1, train_paras["total_iterations"] + 1):
        # Resample training instances every 20 iteration
        if (i - 1) % train_paras["parallel_iterations"] == 0:
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            env = gym.make('fjsp-v0', case=case, env_paras=train_paras)
            print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act_prob(state, memory)
            state, rewards, dones = env.step(actions)
            done = dones.all()
            memory.rewards.append(rewards)
            memory.terminals.append(dones)
        env.reset()

        # Update the policy each iteration
        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memory, train_paras)
            train_rewards.append(reward)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            memory.clear_memory()
            vis.line(X=np.array([i]), Y=np.array([reward]),
                win='window{}'.format(0), update='append', opts=dict(title='Reward'))

        # Validate the policy every 10 iteration
        if i % train_paras["save_timestep"] == 0:
            print('\nStart validating')
        
            vali_result = validate(valid_paras, env_valid, model.policy_old)
            valid_results.append(vali_result.item())

            # Save the best model
            if vali_result < makespan_best:
                makespan_best = vali_result
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)
            vis.line(X=np.array([i]), Y=np.array([vali_result.item()]),
                     win='window{}'.format(2), update='append', opts=dict(title='Validition Makespan'))

    # Save the data of training curve to files
    data = pd.DataFrame(np.array(valid_results).transpose(), columns=["result"])
    data.to_excel(writer_makesapn, sheet_name='Sheet1', index=False, startcol=1)
    writer_makesapn._save()
    writer_makesapn.close()

    data = pd.DataFrame(np.array(train_rewards).transpose(), columns=["reward"])
    data.to_excel(writer_reward, sheet_name='Sheet1', index=False, startcol=1)
    writer_reward._save()
    writer_reward.close()

    print("total_time: ", time.time() - start_time)

if __name__ == '__main__':
    main()