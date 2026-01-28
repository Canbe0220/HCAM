import copy
import json
import os
import random
import time as time

import gym
import pandas as pd
import torch
import numpy as np

import pynvml
from data_utils import nums_detec
import fjsp_env
import ppo

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    setup_seed(500) # 400 50 300
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
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
    test_paras = load_dict["test_paras"]

    train_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(train_paras)
    num_ins = test_paras["num_ins"]
    if test_paras["sample"]:
        env_test_paras["batch_size"] = test_paras["num_sample"]
    else:
        env_test_paras["batch_size"] = 1

    data_path = "./data_test/{0}/".format(test_paras["data_path"])
    test_files = os.listdir(data_path)
    test_files.sort(key=lambda x: x[:-4])
    test_files = test_files[:num_ins]
    mod_files = os.listdir('./model/')[:]

    memory = ppo.Memory()
    model = ppo.PPO(train_paras, model_paras, ppo_paras)
    rules = []
    envs = [] 

    for root, ds, fs in os.walk('./model/'):
        for f in fs:
            if f.endswith('.pt'):
                rules.append(f)

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './test_results/test_{0}'.format(str_time)
    os.makedirs(save_path)

    file_name = [test_files[i] for i in range(num_ins)]
    data_file = pd.DataFrame(file_name, columns=["file_name"])

    writer_makesapn = pd.ExcelWriter('{0}/makespan_{1}.xlsx'.format(save_path, str_time))  # makespan 
    data_file.to_excel(writer_makesapn, sheet_name='Sheet1', index=False)
    writer_makesapn._save()

    writer_time = pd.ExcelWriter('{0}/time_{1}.xlsx'.format(save_path, str_time))  # time 
    data_file.to_excel(writer_time, sheet_name='Sheet1', index=False)
    writer_time._save()

    # Rule-by-rule (model-by-model) testing
    start = time.time()
    for i_rules in range(len(rules)):
        rule = rules[i_rules]
        # Load trained model
        if rule.endswith('.pt'):
            if device.type == 'cuda':
                model_CKPT = torch.load('./model/' + mod_files[i_rules])
            else:
                model_CKPT = torch.load('./model/' + mod_files[i_rules], map_location='cpu')
            print('\nloading checkpoint:', mod_files[i_rules])
            model.policy.load_state_dict(model_CKPT)
            model.policy_old.load_state_dict(model_CKPT)
        print('rule:', rule)

        # Schedule instance by instance
        step_time_last = time.time()
        makespans = []
        times = []
        for i_ins in range(num_ins):
            test_file = data_path + test_files[i_ins]
            with open(test_file) as file_object:
                line = file_object.readlines()
                ins_num_jobs, ins_num_mas, _ = nums_detec(line)
            env_test_paras["num_jobs"] = ins_num_jobs
            env_test_paras["num_mas"] = ins_num_mas

            # Environment object already exists
            if len(envs) == num_ins:
                env = envs[i_ins]
            # Create environment object
            else:
                # Clear the existing environment
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if meminfo.used / meminfo.total > 0.7:
                    envs.clear()
                if test_paras["sample"]:
                    env = gym.make('fjsp-v0', case=[test_file] * test_paras["num_sample"],
                                   env_paras=env_test_paras, data_source='file')
                else:
                    env = gym.make('fjsp-v0', case=[test_file], env_paras=env_test_paras, data_source='file')
                envs.append(copy.deepcopy(env))
                print("Create env[{0}]".format(i_ins))
            
            if test_paras["sample"]:
                makespan, time_re = schedule(env, model, memory, flag_sample=test_paras["sample"])
                makespans.append(torch.min(makespan))
                times.append(time_re)
            else:
                time_s = []
                makespan_s = []  # In fact, the results obtained by DRL-G do not change
                for j in range(test_paras["num_average"]):
                    makespan, time_re = schedule(env, model, memory)
                    makespan_s.append(makespan)
                    time_s.append(time_re)
                    env.reset()
                makespans.append(torch.mean(torch.tensor(makespan_s)))
                times.append(torch.mean(torch.tensor(time_s)))
            print("finish env {0}".format(i_ins))
        print("rule_spend_time: ", time.time() - step_time_last)

        # Save makespan and time data to files
        data = pd.DataFrame(torch.tensor(makespans).t().tolist(), columns=[rule])
        data.to_excel(writer_makesapn, sheet_name='Sheet1', index=False, startcol=i_rules + 1)

        data = pd.DataFrame(torch.tensor(times).t().tolist(), columns=[rule])
        data.to_excel(writer_time, sheet_name='Sheet1', index=False, startcol=i_rules + 1)

        for env in envs:
            env.reset()

    writer_makesapn._save()
    writer_makesapn.close()
    writer_time._save()
    writer_time.close()
    print("total_spend_time: ", time.time() - start)

def schedule(env, model, memory, flag_sample=False):
    # Get state and completion signal
    state = env.state
    dones = env.done_batch
    done = False  # Unfinished at the beginning
    last_time = time.time()
    i = 0
    while ~done:
        i += 1
        with torch.no_grad():
            actions = model.policy_old.act_prob(state, memory, flag_train=False, flag_sample=flag_sample)
        state, rewards, dones = env.step(actions)  # environment transit
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    return copy.deepcopy(env.makespan_batch), spend_time


if __name__ == '__main__':
    main()