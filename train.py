import copy
from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from benchmarl.experiment import ExperimentConfig

# Loads from "benchmarl/conf/experiment/base_experiment.yaml"
experiment_config = ExperimentConfig.get_from_yaml() # We start by loading the defaults

# Override devices
experiment_config.sampling_device = device
experiment_config.train_device = device
experiment_config.buffer_device = device

experiment_config.gamma = 0.99
experiment_config.on_policy_collected_frames_per_batch = 60_000 # Number of frames collected each iteration
experiment_config.on_policy_n_minibatch_iters = 45
experiment_config.on_policy_minibatch_size = 4096
experiment_config.evaluation = True
experiment_config.render = True
experiment_config.share_policy_params = True # Policy parameter sharing on
experiment_config.evaluation_interval = 120_000 # Interval in terms of frames, will evaluate every 120_000 / 60_000 = 2 iterations
experiment_config.evaluation_episodes = 200 # Number of vmas vectorized enviornemnts used in evaluation

#experiment_config.max_n_frames = 6_000 # Runs one iteration, change to 50_000_000 for full training
experiment_config.max_n_frames = 6_000_000 # full training
experiment_config.on_policy_n_envs_per_worker = 600 # Remove this line for full training
#experiment_config.on_policy_n_minibatch_iters = 1 # Remove this line for full training
experiment_config.loggers = ["wandb"] # csv or wandb
experiment_config.checkpoint_interval = 240_000
experiment_config.keep_checkpoints_num = 5


# Loads from "benchmarl/conf/task/vmas/boids.yaml"
task = VmasTask.BOIDS.get_from_yaml()

range = 1.5
world_size = 3
use_influence = False
task.config = {
    "n_agents": 10,
    "max_steps": 400,
    "agent_obs_range": range,
    "world_size_y": world_size,
    "world_size_x": world_size,
    "use_influence": use_influence,
}

from benchmarl.algorithms import MappoConfig

# We can load from "benchmarl/conf/algorithm/mappo.yaml"
algorithm_config = MappoConfig.get_from_yaml()

# Or create it from scratch
algorithm_config = MappoConfig(
        share_param_critic=True, # Critic param sharing on
        clip_epsilon=0.2,
        entropy_coef=0.001, # We modify this, default is 0
        critic_coef=1,
        loss_critic_type="l2",
        lmbda=0.9,
        scale_mapping="biased_softplus_1.0", # Mapping for standard deviation
        use_tanh_normal=True,
        minibatch_advantage=False,
    )

from benchmarl.models import GnnConfig, SequenceModelConfig, MlpConfig
import torch_geometric

gnn_config = GnnConfig(
    topology="from_pos", # Tell the GNN to build topology from positions and edge_radius
    edge_radius=range, # The edge radius for the topology
    self_loops=False,
    gnn_class=torch_geometric.nn.conv.GATv2Conv,
    gnn_kwargs={"add_self_loops": False, "residual": True}, # kwargs of GATv2Conv, residual is helpful in RL
    position_key="pos",
    pos_features=2,
    velocity_key="vel",
    vel_features=2,
    exclude_pos_from_node_features=True, # Do we want to use pos just to build edge features or also keep it in node features? Here we remove it as we want to be invariant to system translations (we do not use absolute positions)
    team_features=1,
    team_key="team",
)
if use_influence:
    gnn_config.influence_key="influence" # edge attributes for influence
    gnn_config.influence_features=1

# We add an MLP layer to process GNN output node embeddings into actions
mlp_config = MlpConfig.get_from_yaml()

# Chain them in a sequence
model_config = SequenceModelConfig(model_configs=[gnn_config, mlp_config], intermediate_sizes=[256])
critic_model_config = MlpConfig.get_from_yaml()

from benchmarl.experiment import Experiment


experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
)
experiment.run()