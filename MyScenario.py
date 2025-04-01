from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
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

# from SenseSphere import SenseSphere
# from BoidDynamics import BoidDynamics
# from Triangle import Triangle

class MyScenario(BaseScenario):
  #####################################################################
  ###                    make_world function                        ###
  #####################################################################
  def make_world(self, batch_dim: int, device: torch.device, **kwargs):

    self.world_size_x = kwargs.pop("world_size_x", 2)
    self.world_size_y = kwargs.pop("world_size_y", 2)
    self.plot_grid = True

    ##############################
    ###### INFO FROM KWARGS ######
    ##############################

    ''' Entity spawning '''
    self.world_spawning_x = kwargs.pop("world_spawning_x", self.world_size_x * 0.9)
    self.world_spawning_y = kwargs.pop("world_spawning_y", self.world_size_y * 0.5)
    self.min_distance_between_entities = (kwargs.pop("agent_radius", 0.1) * 2 + 0.05)

    ''' Agent entities '''
    self.n_agents = kwargs.pop("n_agents", 2)
    self.n_teams = kwargs.pop("teams", 2)

    ''' Goal entities '''
    self.n_goals = kwargs.pop("n_goals", 5)
    self.goal_color = kwargs.pop("goal_colour", Color.YELLOW)
    self.flat_goal_reward = kwargs.pop("flat_goal_reward", 100)
    self.goal_range = kwargs.pop("goal_range", 1.0)
    self.goal_threshold = kwargs.pop("goal_threshold", 2)
    self.goal_respawn = kwargs.pop("goal_respawn", True)

    ''' Reward Info '''
    self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
    self.wall_collision_penalty = kwargs.pop("wall_collision_penalty", -1)
    self.shared_rew = kwargs.pop("shared_rew", False)
    self.min_collision_distance = (0.005)

    ''' Warn if not all kwargs have been consumed '''
    ScenarioUtils.check_kwargs_consumed(kwargs)

    ####################################
    ######### MAKING THE WORLD #########
    ####################################

    world = World(
      batch_dim=1,  # Number of environments
      device=device,  # Use your hardware (GPU/CPU)
      substeps=5,  # Substeps for simulation accuracy
      collision_force=500,  # Collision force for agent interaction
      dt=0.1,  # Simulation timestep
      drag=0.1,  # Optional drag for agent movement
      linear_friction=0.05,  # Optional friction
      angular_friction=0.02,  # Optional angular friction
      x_semidim=self.world_size_x, # bounds of the world
      y_semidim=self.world_size_y, # bounds of the world
    )

    known_colors = [
          Color.GREEN, # Team 1
          Color.RED,    # Team 2
          Color.YELLOW # Rewards
    ]
    self.goals = []

    ''' Adding agents '''
    self.teams = {}
    for team in range(self.n_teams):
      self.teams[team] = []
      for agent_num in range(int(self.n_agents)):

        sensors = [SenseSphere(world)]
        agent = Agent(
          name=f"team_{team}_agent_{agent_num}",
          collide=True,
          rotatable=True,
          color=known_colors[team],
          render_action=True,
          sensors=sensors,
          shape=Triangle(),
          u_range=[1],  # Ranges for actions
          u_multiplier=[1],  # Action multipliers
          dynamics=BoidDynamics(world=world, team=team)
        )

        agent.pos_rew = torch.zeros(
          batch_dim, device=device
        )  # Tensor that will hold the position reward fo the agent
        agent.agent_collision_rew = (
          agent.pos_rew.clone()
        )  # Tensor that will hold the collision reward fo the agent

        self.teams[team].append(agent)
        world.add_agent(agent)

    ''' Adding Goals  '''
    for i in range(self.n_goals):
      goal = Landmark(
        name=f"goal_{i}",
        collide=True,
        movable = False,
        color=known_colors[2],
      )
      goal.range = self.goal_range
      goal.threshold = self.goal_threshold
      self.goals.append(goal)
      world.add_landmark(goal)

    return world

  #####################################################
  ############## reset_world_at function ##############
  #####################################################
  def reset_world_at(self, env_index: int = None):
    # Spawn friendlies
    ScenarioUtils.spawn_entities_randomly(
      self.teams[0],
      self.world,
      env_index, # Pass the env_index so we only reset what needs resetting
      self.min_distance_between_entities,
      x_bounds=(-self.world_size_x, self.world_size_x),
      y_bounds=(-self.world_size_y, -self.world_size_y),
    )
    # Spawn Enemies
    ScenarioUtils.spawn_entities_randomly(
      self.teams[1],
      self.world,
      env_index, # Pass the env_index so we only reset what needs resetting
      self.min_distance_between_entities,
      x_bounds=(-self.world_size_x, self.world_size_x),
      y_bounds=(self.world_size_y, self.world_size_y),
    )
    # Spawn Goals
    ScenarioUtils.spawn_entities_randomly(
      self.goals,  # List of entities to spawn
      self.world,
      env_index, # Pass the env_index so we only reset what needs resetting
      self.min_distance_between_entities,
      x_bounds=(-self.world_spawning_x, self.world_spawning_x),
      y_bounds=(-self.world_spawning_y, self.world_spawning_y),
    )

    for agent in self.world.agents:
      agent.dynamics.reset(0)

  ##################################################
  ############## observation function ##############
  ##################################################
  def observation(self, agent: Agent):
    ''' Measure observations from the agent's sensors '''
    sensor_observations = [sensor.measure() for sensor in agent.sensors]
    ''' Flatten and concatenate sensor data into a single tensor '''
    obs = {
      "obs": torch.cat(sensor_observations, dim=-1),
      "pos": agent.state.pos,
      "vel": agent.state.vel,
    }
    return obs

  #############################################
  ############## reward function ##############
  #############################################
  def reward(self, agent: Agent):
    is_first = agent == self.world.agents[0]
    is_last = agent == self.world.agents[-1]

    ''' Taken from the discovery.py scenario '''
    goal_reward = 0
    if is_first:
      pass
      ''' negative reward for time passing - don't think it's relevant for BOIDS '''
      # self.time_rew = torch.full(
      #   (self.world.batch_dim,),
      #   self.time_penalty,
      #   device=self.world.device,
      # )

      ''' updating tensor of all agent positions - shape [board_dimensions, num_agents] '''
      self.agents_pos = torch.stack(
        [a.state.pos for a in self.world.agents], dim=1
      )
      ''' updating tensor of all goal positions '''
      self.goals_pos = torch.stack([g.state.pos for g in self.goals], dim=1)

      ''' getting tensor with distances between reward positions and agent positions '''
      self.agents_goals_dists = torch.cdist(self.agents_pos, self.goals_pos)

      self.agents_per_goal = torch.sum(
        (self.agents_goals_dists < self.goal_range).type(torch.int),
        dim=1,
      )

      self.covered_goals = self.agents_per_goal >= self.goal_threshold

      ''' Flat reward for each goal captured by the team '''
      goal_reward = 0
      # print(self.covered_goals)
      # print(self.covered_goals.shape())
      for goal_covered in self.covered_goals[0]:
        if goal_covered:
            goal_reward += self.flat_goal_reward  # Add flat reward for each goal covered

    if is_last:
      if self.goal_respawn:
        occupied_positions_agents = [self.agents_pos]
        for i, goal in enumerate(self.goals):
          occupied_positions_goals = [
            o.state.pos.unsqueeze(1)
            for o in self.goals
            if o is not goal
          ]
          occupied_positions = torch.cat(
            occupied_positions_agents + occupied_positions_goals,
            dim=1,
          )
          pos = ScenarioUtils.find_random_pos_for_entity(
            occupied_positions,
            env_index=None,
            world=self.world,
            min_dist_between_entities=self.min_distance_between_entities,
            x_bounds=(-self.world.x_semidim, self.world.x_semidim),
            y_bounds=(-self.world.y_semidim, self.world.y_semidim),
          )

          goal.state.pos[self.covered_goals[:, i]] = pos[
            self.covered_goals[:, i]
          ].squeeze(1)
      else:
        self.all_time_covered_goals += self.covered_goals
        for i, goal in enumerate(self.goals):
          goal.state.pos[self.covered_goals[:, i]] = self.get_outside_pos(
            None
          )[self.covered_goals[:, i]]

    ''' Negative reward for touching boundaries (walls) '''
    coll_pen = 0
    pos_x = agent.state.pos[0][0]
    pos_y = agent.state.pos[0][1]
    if abs(pos_x) == self.world_size_x or abs(pos_y) == self.world_size_y:
      coll_pen = -1  # Apply negative penalty for wall collisions

    ''' Combining goal reward and collision penalty '''
    reward = goal_reward + coll_pen

    ''' Return the total reward (tensor of shape [self.world.batch_dim,]) '''
    return torch.tensor([reward], device=self.world.device)

  #############################################
  ############## Extra_render ################
  #############################################
  def extra_render(self, env_index: int = 0) -> "List[Geom]":
    from vmas.simulator import rendering

    geoms: List[Geom] = []

    # Goal covering ranges
    for goal in self.goals:
      range_circle = rendering.make_circle(goal.range, filled=False)
      xform = rendering.Transform()
      xform.set_translation(*goal.state.pos[env_index])
      range_circle.add_attr(xform)
      range_circle.set_color(*self.goal_color.value)
      geoms.append(range_circle)

    return geoms
