# MARL_boids
A Multi-Agent Reinforcement Learning project evaluating the impact of learnable communication/influence in a boids-like swarm v swarm king-of-the-hill game.

## Installation
For Linux:
```
git clone https://github.com/AdamKoziak1/MARL_boids.git
cd MARL_boids/
python -m venv venv
source venv/bin/activate
pip install -e VectorizedMultiAgentSimulator
pip install -e BenchMARL
pip install -r requirements.txt
```
Note, torch_cluster takes a while to install. 

## Play the Environment
```
python play_interactively.py
```
Control two agents with a,d or left/right arrow keys

## Train the Policy
```
python train.py
```
Then, a directory will be created with logs, videos, etc added automatically as training progresses.
