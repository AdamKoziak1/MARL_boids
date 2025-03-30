import torch
from torch import Tensor
from vmas import make_env, render_interactively

from MyScenario import MyScenario

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env = make_env(
#     MyScenario(),
#     num_envs=3,
#     #num_agents=2,
# )
# print(env.reset())

render_interactively(
    MyScenario(),
    #"discovery",
    control_two_agents=True,
    save_render=False,
    display_info=True,
)