import torch
from vmas.simulator.utils import ScenarioUtils
from vmas.simulator.core import World
from torch import Tensor
from vmas import make_env, render_interactively

from MyScenario import MyScenario
#from example_scenario import MyScenario

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env = make_env(
#     MyScenario(),
#     num_envs=3,
#     #num_agents=2,
# )
# print(env.reset())

render_interactively(
    MyScenario(),
    control_two_agents=True,
    save_render=False,
    display_info=True,
)