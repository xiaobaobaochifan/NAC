# README

This is the official repository for the algorithm of Net Actor-Critic (NAC) and reproducing the results of the paper titled [To Switch or Not to Switch? Balanced Policy Switching in Offline Reinforcement Learning](https://arxiv.org/abs/2407.01837). 

There are two folders inside `src`:
* `continuous`: complete set of source code for continuous state space cases (i.e. Gymnasium);
* `discrete`: complete set of source code for discrete finite state space cases (i.e. SUMO).

The content of each folder/script within either `continuous` or `discrete`: 
* `algs`: implementation of the proposed Net Actor-Critic algorithm;
* `utils`: logger function, pre-trained suboptimal/optimal old policies in different environments; 
* `main.py` or `main_discrete.py`: main script to run both experiments and testings;
* `main_ol.py` or `main_ol_discrete.py`: the script to train an optimal policy online.


In the following sections, the 'Requirements' lists all necessary packages and their versions. 'Environment Configuration' guides you through setting up the execution environment, and 'Execution' provides an example command to run the code.

## Requirements

To execute the provided code, the following dependencies are necessary.  

```
python == 3.11.9
gymnasium == 0.29.1
joblib
numpy
torch == 2.3.0
swig
sumo-rl == 1.4.5
```

## Environment configuration
Here we provide the setup of the environment step by step. To avoid discrepancies, it is recommended to set separate environment for continuous and discrete experiments.

Continuous case:

```
conda create -n gymrl python=3.11
conda activate gymrl

pip install mujoco
pip install "gymnasium[all]"
conda install -c conda-forge xorg-libx11
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
nano ~/.bashrc
# Add the following line to the .bashrc file. 
export CPATH=$CONDA_PREFIX/include
pip install patchelf
pip install joblib

# Installation of mujoco 2.1.0:
sudo mkdir ~/.mujoco
tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
nano .bashrc
# Add the following line to .bashrc, please replace the user-name as your user name. 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user-name/.mujoco/mujoco210/bin
source ~/.bashrc

# Verify the installation of mujoco 2.1.0:
cd ~/.mujoco/mujoco210/bin
./simulate ../model/humanoid.xml
```

Discrete case:

```
conda create -n sumorl python=3.11
conda activate sumorl

sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc

pip install sumo-rl
pip install joblib
```

## Execution

After we setup the environment successfully, please run the following to execute the code:
```
cd continuous
python main.py --reps 10 --exp_name 'try' --epochs_trn 100 --epochs_eva 50 --seed 4
```

To reproduce the results claimed in the paper, please just setup the corresponding parameters as described in the paper. 

## Citation
If you use this repository or the proposed methods in your work, please cite it as follows:

```bibtex
@article{ma2024switch,
  title={To switch or not to switch? Balanced policy switching in offline reinforcement learning},
  author={Ma, Tao and Yang, Xuzhi and Szabo, Zoltan},
  journal={arXiv preprint arXiv:2407.01837},
  year={2024}
}
```
