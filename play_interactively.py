import torch
from vmas import render_interactively

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

render_interactively(
    "boids",
    control_two_agents=True,
    save_render=False,
    display_info=True,
)
