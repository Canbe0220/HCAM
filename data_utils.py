import torch
import numpy as np
import random
import time
import json

class CaseGenerator:
    '''
    FJSP instance generator
    '''
    def __init__(self, job_init, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=None, path="data/", flag_same_opes=True, flag_doc=False):
        if nums_ope is None:
            nums_ope = []
        self.flag_doc = flag_doc  # Whether save the instance to a file
        self.flag_same_opes = flag_same_opes
        self.nums_ope = nums_ope
        self.path = path  # Instance save path (relative path)
        self.job_init = job_init
        self.num_mas = num_mas

        self.mas_per_ope_min = 1  # The minimum number of machines that can process an operation
        self.mas_per_ope_max = num_mas
        self.opes_per_job_min = opes_per_job_min  # The minimum number of operations for a job
        self.opes_per_job_max = opes_per_job_max
        self.proctime_per_ope_min = 1  # Minimum average processing time
        self.proctime_per_ope_max = 20
        self.proctime_dev = 0.2

    def get_case(self, idx=0):
        '''
        Generate FJSP instance
        :param idx: The instance number
        '''
        self.num_jobs = self.job_init
        if not self.flag_same_opes:
            self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)]
        self.num_opes = sum(self.nums_ope)
        self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]
        self.num_options = sum(self.nums_option)
        self.ope_ma = []
        for val in self.nums_option:
            self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_mas+1), val))
        self.proc_time = []
        self.proc_times_mean = [random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max) for _ in range(self.num_opes)]
        for i in range(len(self.nums_option)):
            low_bound = max(self.proctime_per_ope_min,round(self.proc_times_mean[i]*(1-self.proctime_dev)))
            high_bound = min(self.proctime_per_ope_max,round(self.proc_times_mean[i]*(1+self.proctime_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
            self.proc_time = self.proc_time + proc_time_ope
        self.num_ope_biass = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        self.num_ma_biass = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes)
        lines = []
        lines_doc = []
        lines.append(line0)
        lines_doc.append('{0}\t{1}\t{2}'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes))
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1
            idx_ma = 0
            line = []
            option_max = sum(self.nums_option[self.num_ope_biass[i]:(self.num_ope_biass[i]+self.nums_ope[i])])
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
                    flag_new_ope += self.nums_option[self.num_ope_biass[i]+idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_biass[i]+idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.proc_time[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)
                    break
        lines.append('\n')
        if self.flag_doc:
            doc = open(self.path + '{0}j_{1}m_{2}.fjs'.format(self.num_jobs, self.num_mas, str.zfill(str(idx+1),3)),'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            doc.close()
        return lines, self.num_jobs, self.num_jobs


def load_fjs(lines, num_mas, num_opes):
    '''
    Load the local FJSP instance.
    '''
    flag = 0
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    nums_ope = []  # A list of the number of operations for each job
    num_ope_biases = []  # The id of the first operation of each job
    # Parse data line by line
    for line in lines:
        # first line
        if flag == 0:
            flag += 1
        # last line
        elif line is "\n":
            break
        # other
        else:
            num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
            num_ope_biases.append(num_ope_bias)
            # Detect information of this job and return the number of operations
            num_ope = edge_detec(line, num_ope_bias, matrix_proc_time)
            nums_ope.append(num_ope)
            # nums_option = np.concatenate((nums_option, num_option))
            flag += 1
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
    return matrix_proc_time, matrix_ope_ma_adj, torch.tensor(num_ope_biases).int(), torch.tensor(nums_ope).int()

def nums_detec(lines):
    '''
    Count the number of jobs, machines and operations
    '''
    num_opes = 0
    for i in range(1, len(lines)):
        num_opes += int(lines[i].strip().split()[0]) if lines[i]!="\n" else 0
    line_split = lines[0].strip().split()
    num_jobs = int(line_split[0])
    num_mas = int(line_split[1])
    return num_jobs, num_mas, num_opes

def edge_detec(line, num_ope_bias, matrix_proc_time):
    '''
    Detect information of a job
    '''
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0  # Store the number of operations of this job
    num_option = np.array([])  # Store the number of processable machines for each operation of this job
    mac = 0
    for i in line_split:
        x = int(i)
        # The first number indicates the number of operations of this job
        if flag == 0:
            num_ope = x
            flag += 1
        # new operation detected
        elif flag == flag_new_ope:
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            flag += 1
        # not proc_time (machine)
        elif flag_time == 0:
            mac = x-1
            flag += 1
            flag_time = 1
        # proc_time
        else:
            matrix_proc_time[idx_ope+num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
    return num_ope

def read_json(path:str) -> dict:
    with open(path+".json","r",encoding="utf-8") as f:
        config = json.load(f)
    return config

def write_json(data:dict, path:str):
    with open(path+".json", 'w', encoding='UTF-8') as fp:
        fp.write(json.dumps(data, indent=2, ensure_ascii=False))