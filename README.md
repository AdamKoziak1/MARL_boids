# MARL_boids

## Installation
For Linux:
```
python -m venv venv
source venv/bin/activate
pip install -e VectorizedMultiAgentSimulator
pip install -e BenchMARL
pip install -r requirements.txt
```

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
