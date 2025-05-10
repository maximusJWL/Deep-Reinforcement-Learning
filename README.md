# Deep-Reinforcement-Learning

This repo is for a deep reinforcement learning project on the Taxi-v3 and ALE Boxing-v5 environments from Gymnasium.
The algorithms used in the project are: Q-Learning, Duelling DQN with Prioritised Experience Replay, Proximal Policy Optimisation.


"Taxi_DQN.ipynb" is the notebook for the training and evaluation of the DQN, with visualisations.
"Taxi QL.ipynb" is the notebook for the training and evaluation of the QL algorithms, with visualisations.

These can be run in google colab out of the box as of 10/05/2025.

"ALE Boxing training.py" is the training script for the PPO algorithm.
"ALE Boxing demo.py" is the script for showing the agent playing the game (agent loads from a checkpoint file).
The algorithm can be loaded from the checkpoint zip file to save you needing to train it yourself. 

Required packages and versions:
Python ---------- 3.11.0
Gymnasium ------- 1.0.0
Raylib ---------- 2.45.0
Pandas ---------- 2.2.3
Torch ----------- 2.7.0

full list (maybe unnecessary but listed in case of phantom dependencies):
absl-py                   2.2.2      
ale-py                    0.11.0     
attrs                     25.3.0     
certifi                   2025.4.26  
charset-normalizer        3.4.2      
click                     8.1.8      
cloudpickle               3.1.1      
colorama                  0.4.6      
dm-tree                   0.1.9      
Farama-Notifications      0.0.4      
filelock                  3.18.0     
fsspec                    2025.3.2   
gymnasium                 1.0.0      
idna                      3.10       
Jinja2                    3.1.6      
jsonschema                4.23.0     
jsonschema-specifications 2025.4.1   
lz4                       4.4.4
MarkupSafe                3.0.2
mpmath                    1.3.0
msgpack                   1.1.0
networkx                  3.4.2
numpy                     2.2.5
opencv-python             4.11.0.86
ormsgpack                 1.7.0
packaging                 25.0
pandas                    2.2.3
pillow                    11.2.1
pip                       22.3
protobuf                  6.30.2
pyarrow                   20.0.0
python-dateutil           2.9.0.post0
pytz                      2025.2
PyYAML                    6.0.2
ray                       2.45.0
referencing               0.36.2
requests                  2.32.3
rpds-py                   0.24.0
scipy                     1.15.2
setuptools                65.5.0
six                       1.17.0
sympy                     1.14.0
tensorboardX              2.6.2.2
torch                     2.7.0
typing_extensions         4.13.2
tzdata                    2025.2
urllib3                   2.4.0
wrapt                     1.17.2
