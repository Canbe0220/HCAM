import numpy as np
import random
import torch
from data_utils import CaseGenerator

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Generate instances and save to files
def main():
    setup_seed(100)
    batch_size = 100
    num_jobs = 20
    num_mas = 10
    opes_per_job_min = int(num_mas)
    opes_per_job_max = int(num_mas)
    nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
    case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope, path='data_val/2010/', flag_doc=True)
    for i in range(batch_size):
        case.get_case(i)

if __name__ == "__main__":
    main()
    print("Data_val is created. ")