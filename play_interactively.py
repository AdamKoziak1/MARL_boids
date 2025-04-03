import torch
from torch import Tensor
from vmas import render_interactively

#from boids import Scenario

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

render_interactively(
    "boids", # for when vmas is set up with the boids env, "discovery" or "football" for default envs
    #Scenario(), # otherwise
    control_two_agents=True,
    save_render=False,
    display_info=True,
)
