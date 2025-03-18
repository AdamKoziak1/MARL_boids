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
    self.n_friends = kwargs.pop("friends", 5)
    self.n_enemies = kwargs.pop("enemies", 5)
    self.n_agents = self.n_friends + self.n_enemies
    self.n_teams = kwargs.pop("teams", 2)

    ''' Number of non-agent entities '''
    self.n_obstacles = kwargs.pop("n_obstacles", 2)
    self.n_goals = kwargs.pop("n_obstacles", 3)

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
          Color.YELLOW
    ]

    self.goals = []

    ''' Add agents '''
    teams = {}
    for team in range(self.n_teams):
      teams[team] = []
      for agent_num in range(int(self.n_agents//self.n_teams)):

        agent = BoidAgent(
          collide=True,
          color=known_colors[team],
          render_action=True,
          position=1,
          velocity=1,
          sensor_range=1.0,
          max_speed=1.0,
          max_force=0.1
        )

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
  def observation(agent, world):
    """
    Collect the relevant observation data for the given agent from the world.
    Only consider other Boids within the agent's sensor range.
    """
    obs = {
      "pos": agent.position,
      "vel": agent.velocity,
    }

    # Example: Relative positions and velocities of nearby agents (simple flocking)
    for other_agent in world.agents:
      if other_agent != agent:
        # Calculate the distance between agents
        distance = torch.norm(agent.position - other_agent.position)
        if distance <= agent.sensor_range:  # Check if within the sensor range
          # Relative position and velocity of nearby Boid
          relative_pos = other_agent.position - agent.position
          relative_vel = other_agent.velocity - agent.velocity
          obs[f"relative_pos_{id(other_agent)}"] = relative_pos
          obs[f"relative_vel_{id(other_agent)}"] = relative_vel

  # Optionally add more observations (like obstacles, goal positions, etc.)
    return obs

  #####################################################################
  ###                       reward function                         ###
  #####################################################################
  def reward(self, agent: BoidAgent):
    return agent.captured_objects  # Assuming this is tracked elsewhere
