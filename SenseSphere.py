import torch
from vmas.simulator import rendering
from vmas.simulator.core import Sensor
from typing import List
from vmas.simulator.rendering import Geom

class SenseSphere(Sensor):
  def __init__(self, world, range=1.0):
    super().__init__(world)
    self.range = range
    self._last_measurement = None

  def measure(self):
    agent_pos = self.agent.state.pos  # Current agent's position
    agent_vel = self.agent.state.vel  # Current agent's velocity

    observations = []
    for other_agent in self._world.agents:
      if other_agent is self.agent:
        continue  # Skip self

      # Compute relative position and velocity
      rel_pos = other_agent.state.pos - agent_pos
      rel_vel = other_agent.state.vel - agent_vel
      distance = torch.norm(rel_pos, dim=-1)
      # Only include agents within range
      if distance <= self.range:
        observations.append(torch.concat([rel_pos, rel_vel], axis=1))
      else:
        observations.append(torch.zeros((1,4)))
    return torch.stack(observations)


  def render(self, env_index: int = 0) -> "List[Geom]":
    # if not self._render:
    #     return []

    geoms: List[rendering.Geom] = []

    # Render the range of the SenseSphere as a circle around each agent
    circle = rendering.make_circle(radius=self.range)  # Create the sensor's circle based on range
    circle.set_color(0, 0, 1, alpha=0.05)  # Set the color to blue with transparency
    xform = rendering.Transform()
    xform.set_translation(*self.agent.state.pos[env_index])  # Position the circle at the agent's position
    circle.add_attr(xform)

    geoms.append(circle)
    return geoms

  def to(self, device: torch.device):
    self.range = torch.tensor(self.range, device=device)  # Ensure range is a tensor
