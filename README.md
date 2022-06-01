# RLAdmm

An reinforcement learning approach to updating the penalty parameters of alternating direction method of multiplier (ADMM) method.

## Installing dependencies
The code has been tested under Python 3.6. Run the following commands to install the required Python packages.

First, download PyTorch (https://pytorch.org/get-started/locally/). Skip this step if PyTorch is already installed.
```
pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
```
Install the rest of the packages.
```
pip install -r requirements.txt
```

### ExaComDecOPF solver

The current implementation is based on an old ADMM solver `ExaComDecOPF` with some minor modification for the purpose of training the RL model.
This would better be replaced by `ExaAdmm.jl` later.

## Training a policy
Training can be started by running
```
python train_dqn_entrywise.py --num_case 9 --max_episodes 1000 --max_iters 3000 --use_baseline
```

## Publication

- Sihan Zeng, Alyssa Kody, Youngdae Kim, Kibaek Kim, and Daniel K Molzahn. "A Reinforcement Learning Approach to Parameter Selection for Distributed Optimization in Power Systems" arXiv preprint arXiv:2110.11991, 2021 (to appear in PSCC 2022)

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
