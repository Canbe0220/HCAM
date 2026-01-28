# RLEGA-FJSP

This repository is the implementation of the paper “HCAM: Hierarchical Coupling Attentive Modulation via Deep Reinforcement Learning for Flexible Job Shop Scheduling”.

## Get Started

### requirements

* ubuntu $\ge$ 24.04.2
* python $\ge$ 3.8.20
* pytorch $\ge$ 2.1.1
* gym $\ge$ 0.18.0
* numpy $\ge$ 1.19.5
* visdom $\ge$ 0.1.8.9
* pandas $\ge$ 1.3.5

### Introduction

* `data_test` saves all testing public becnhmarks.
* `data_val` saves 100 validation instances.
* `model` saves the trained models for testing. 
* `test_results` saves the testing results solved by RLEGA.
* `train_results` saves the training results.
* `data_create.py` is used for generating synthetic instances.
* `data_utils.py` contains helper functions for generating and loading data.
* `fjsp_env.py` is the implementation of FJSP environment.
* `HCAM.py` contains the SEAM and GRAM modules and decision-making networks.
* `mlp.py` is the MLP code of the actor and critic (referenced from L2D).
* `param.json` is the parameter file.
* `ppo.py` is the implementation of PPO algorithm.
* `test.py` is used for testing.
* `train.py` is used for training.


### train

```
python train.py
```

Note that this file contains training and validating.

### test

```
python test.py
```
Note that model files end with (`*.pt`).

## Reference

* https://github.com/songwenas12/fjsp-drl
* https://github.com/wrqccc/FJSP-DRL
* https://github.com/zcaicaros/L2D