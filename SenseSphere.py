import torch
from vmas.simulator.core import Sensor
from vmas.simulator.rendering import make_circle
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
        observations.append(torch.cat([rel_pos, rel_vel]))

    if observations:
      self._last_measurement = observations
      return torch.stack(observations)
    else:
      self._last_measurement = torch.zeros((0, 4))
      return torch.zeros((0, 4))  # Empty tensor if no neighbors


  def render(self, env_index: int = 0) -> "List[Geom]":
    # if not self._render:
    #     return []
    from vmas.simulator import rendering

    geoms: List[rendering.Geom] = []

    if self._last_measurement is not None:
        # Render the range of the SenseSphere as a circle around each agent
        circle = rendering.make_circle(radius=self.range)  # Create the sensor's circle based on range
        circle.set_color(0, 0, 1, alpha=0.3)  # Set the color to blue with transparency
        xform = rendering.Transform()
        xform.set_translation(*self.agent.state.pos[env_index])  # Position the circle at the agent's position
        circle.add_attr(xform)

        geoms.append(circle)

    return geoms


  # def render(self, env_index: int = 0) -> "List[Geom]":
  #   from vmas.simulator import rendering

  #   geoms: List[rendering.Geom] = []
  #   if self._last_measurement is not None:
  #     for angle, dist in zip(self._angles.unbind(1), self._last_measurement.unbind(1)):
  #       # Make sure each sensor is positioned relative to its agent
  #       angle = angle[env_index] + self.agent.state.rot.squeeze(-1)[env_index]
  #       ray = rendering.Line(
  #           (0, 0),
  #           (dist[env_index], 0),
  #           width=0.05,
  #       )
  #       xform = rendering.Transform()
  #       xform.set_translation(*self.agent.state.pos[env_index])  # Position tied to agent
  #       xform.set_rotation(angle)
  #       ray.add_attr(xform)
  #       ray.set_color(r=0, g=0, b=0, alpha=self.alpha)

  #       # Position for sensor circle relative to agent's position and direction
  #       ray_circ = rendering.make_circle(0.01)
  #       ray_circ.set_color(*self.render_color, alpha=self.alpha)
  #       xform = rendering.Transform()
  #       rot = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
  #       pos_circ = (
  #           self.agent.state.pos[env_index] + rot * dist.unsqueeze(1)[env_index]
  #       )  # This ensures sensor position is relative to agent's position
  #       xform.set_translation(*pos_circ)
  #       ray_circ.add_attr(xform)

  #       geoms.append(ray)
  #       geoms.append(ray_circ)
  #   return geoms


  # def render(self, env_index: int = 0) -> "List[Geom]":
  #   from vmas.simulator import rendering

  #   geoms: List[rendering.Geom] = []

  #   # Create the circle (sensor range) as before
  #   circle = rendering.make_circle(radius=self.range)
  #   circle.set_color(0, 0, 1, alpha=0.3)  # Light blue with transparency

  #   # Create a Transform object to set the position
  #   xform = rendering.Transform()
  #   xform.set_translation(*self.agent.state.pos[env_index])  # Position the circle

  #   # Apply the transformation to the circle
  #   circle.add_attr(xform)

  #   geoms.append(circle)
  #   return geoms

  # def render(self, env_index: int = 0) -> "List[Geom]":

  #   from vmas.simulator import rendering

  #   geoms: List[rendering.Geom] = []
  #   circle = make_circle(radius=self.range)
  #   circle.set_color(0, 0, 1, alpha=0.3)  # Light blue with transparency
  #   circle.set_position(self.agent.state.pos[env_index])


  #   geoms.append(circle)
  #   return geoms

  def to(self, device: torch.device):
    self.range = torch.tensor(self.range, device=device)  # Ensure range is a tensor
