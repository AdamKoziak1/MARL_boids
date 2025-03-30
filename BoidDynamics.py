import torch
import math
from vmas.simulator.dynamics.common import Dynamics

class BoidDynamics(Dynamics):
    def __init__(self, world, constant_speed=0.1, max_steering_rate=0.2*math.pi, team=0):
        super().__init__()
        self.constant_speed = constant_speed
        self.max_steering_rate = max_steering_rate  # max radians per second
        self.world = world
        self.team = team

    @property
    def needed_action_size(self):
        return 1  # Steering input only

    def reset(self, env_index: int):
        if self.team == 0:
            self._agent.state.rot += 0.5*math.pi
        elif self.team == 1:
            self._agent.state.rot -= 0.5*math.pi
        self._agent.state.rot = (self._agent.state.rot + math.pi) % (2 * math.pi) - math.pi
        self._agent.state.vel = self.constant_speed * torch.cat(
            [
                torch.cos(self._agent.state.rot),
                torch.sin(self._agent.state.rot)
            ],
            dim=1
        )
        self._agent.state.force = torch.zeros_like(self._agent.state.force)

    def zero_grad(self):
        pass

    def clone(self):
        return BoidDynamics(self.world, self.constant_speed, self.max_steering_rate)
    
    def process_action(self):
        dt = self.world.dt
        steering_rate = self._agent.action.u[:, 0].clamp(-1, 1) * self.max_steering_rate
        # Update orientation
        self._agent.state.rot -= steering_rate.unsqueeze(1) * dt
        self._agent.state.rot = (self._agent.state.rot + math.pi) % (2 * math.pi) - math.pi

        # Update velocity based on orientation
        self._agent.state.vel = self.constant_speed * torch.cat(
            [
                torch.cos(self._agent.state.rot),
                torch.sin(self._agent.state.rot)
            ],
            dim=1
        )

        # Set force to zero to avoid external forces
        self._agent.state.force = torch.zeros_like(self._agent.state.force)