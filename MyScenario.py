from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
!pip install vmas
!apt-get update
!apt-get install -y x11-utils python3-opengl xvfb
!pip install pyvirtualdisplay
import pyvirtualdisplay
display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
display.start()
from vmas.simulator.scenario import BaseScenario
import torch
from torch import Tensor
from vmas.simulator.core import (
  Agent,
  Box,
  Landmark,
  Sphere,
  World
)
from vmas.simulator.utils import (
  ScenarioUtils,
  Color
)

class MyScenario(BaseScenario):

  #####################################################################
  ###                    make_world function                        ###
  #####################################################################
  def make_world(self, batch_dim: int, device: torch.device, **kwargs):
    self.plot_grid = True

    ''' INFO FROM KWARGS '''
    ''' Number of agents '''
    # self.n_friends = kwargs.pop("friends", 5)
    # self.n_enemies = kwargs.pop("enemies", 5)
    # self.n_agents = self.n_friends + self.n_enemies
    self.n_agents = kwargs.pop("n_agents", 5)
    self.n_teams = kwargs.pop("teams", 2)

    ''' Number of non-agent entities '''
    # self.n_obstacles = kwargs.pop("n_obstacles", 2)
    self.n_goals = kwargs.pop("n_goals", 3)

    ''' cordinates for entities spawning '''
    self.world_spawning_x = kwargs.pop("world_spawning_x", 1)
    self.world_spawning_y = kwargs.pop("world_spawning_y", 1)

    ''' REWARD INFO '''
    ''' Penalize collision with another agent or obstacle '''
    self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
    ''' Turning shared reward on '''
    self.shared_rew = kwargs.pop("shared_rew", False)

    ''' Minimum distance between entities at spawning time '''
    self.min_distance_between_entities = (kwargs.pop("agent_radius", 0.1) * 2 + 0.05)
    ''' Minimum distance between entities for collision trigger '''
    self.min_collision_distance = (0.005)

    ''' Warn if not all kwargs have been consumed '''
    ScenarioUtils.check_kwargs_consumed(kwargs)


    '''*********** MAKING THE WORLD ***********'''

    world = World(
      batch_dim=1,  # Number of environments
      device=device,  # Use your hardware (GPU/CPU)
      substeps=5,  # Substeps for simulation accuracy
      collision_force=500,  # Collision force for agent interaction
      dt=0.1,  # Simulation timestep
      drag=0.1,  # Optional drag for agent movement
      linear_friction=0.05,  # Optional friction
      angular_friction=0.02,  # Optional angular friction
    )

    known_colors = [
          Color.GREEN, # Team 1
          Color.RED,    # Team 2
          Color.YELLOW # Rewards
    ]

    self.goals = []

    # sensors = [SenseSphere(world)]

    ''' Add agents '''
    teams = {}
    for team in range(self.n_teams):
      teams[team] = []
      for agent_num in range(int(self.n_agents)):

        sensors = [SenseSphere(world)]
        agent = Agent(
          name=f"team_{team}_agent_{agent_num}",
          collide=True,
          color=known_colors[team],
          render_action=True,
          sensors=sensors,
          shape=Sphere(radius=0.1),
          u_range=[1, 1],  # Ranges for actions
          u_multiplier=[1, 1],  # Action multipliers
          dynamics=BoidDynamics()  # If you got to its class you can see it has 2 actions: force_x, and force_y
        )

        # agent = BoidAgent(
        #   # name=f"team_{team}_agent_{agent_num}",
        #   name="michael",
        #   sensors=sensors,
        #   collide=True,
        #   color=known_colors[team],
        #   render_action=True,
        #   position=1,
        #   velocity=1,
        #   sensor_range=1.0,
        #   max_speed=1.0,
        #   max_force=0.1
        # )

        agent.pos_rew = torch.zeros(
          batch_dim, device=device
        )  # Tensor that will hold the position reward fo the agent
        agent.agent_collision_rew = (
          agent.pos_rew.clone()
        )  # Tensor that will hold the collision reward fo the agent

        teams[team].append(agent)
        world.add_agent(agent)

    ''' Add Goals '''

    for i in range(self.n_goals):
      goal = Landmark(
        name=f"goal_{i}",
        collide=False,
        color=known_colors[2],
      )
      world.add_landmark(goal)
      # agent.goal = goal
      self.goals.append(goal)

    ''' Add later... '''

    return world

  #####################################################################
  ###                  reset_world_at function                      ###
  #####################################################################
  def reset_world_at(self, env_index: int = None):
    ScenarioUtils.spawn_entities_randomly(
      self.world.agents
      # + self.obstacles # no obstacles at the moment
      + self.goals,  # List of entities to spawn
      self.world,
      env_index, # Pass the env_index so we only reset what needs resetting
      self.min_distance_between_entities,
      x_bounds=(-self.world_spawning_x, self.world_spawning_x),
      y_bounds=(-self.world_spawning_y, self.world_spawning_y),
    )

  #####################################################################
  ###                    observation function                       ###
  #####################################################################
  def observation(self, agent: Agent):
    # Measure observations from the agent's sensors
    sensor_observations = [sensor.measure() for sensor in agent.sensors]

    # Flatten and concatenate sensor data into a single tensor
    obs = {
        "obs": torch.cat(sensor_observations, dim=-1),
        "pos": agent.state.pos,
        "vel": agent.state.vel,
    }

    return obs


  #####################################################################
  ###                       reward function                         ###
  #####################################################################
  def reward(self, agent: Agent):
    # Dummy reward: Just return 1 for every call to the reward function
    return torch.tensor(1.0)
