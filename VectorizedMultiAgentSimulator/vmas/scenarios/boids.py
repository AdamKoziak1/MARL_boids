import math
import numpy as np
from vmas.simulator import rendering
from vmas.simulator.core import Agent, Landmark, World, Shape, Sensor
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.rendering import Geom
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils, Color
import torch
from torch import Tensor
from typing import List


class Scenario(BaseScenario):
  #####################################################################
  ###                    make_world function                        ###
  #####################################################################
  def make_world(self, batch_dim: int, device: torch.device, **kwargs):
    self.batch_dim = batch_dim
    self.device=device
    self.world_size_x = kwargs.pop("world_size_x", 3)
    self.world_size_y = kwargs.pop("world_size_y", 3)
    self.plot_grid = True
    self.agent_obs_range = kwargs.pop("agent_obs_range", 1.5)
    self.use_influence = kwargs.pop("use_influence", True)
    print(self.use_influence)

    ##############################
    ###### INFO FROM KWARGS ######
    ##############################

    ''' Entity spawning '''
    self.world_spawning_x = kwargs.pop("world_spawning_x", self.world_size_x * 0.9)
    self.world_spawning_y = kwargs.pop("world_spawning_y", self.world_size_y * 0.5)
    self.min_distance_between_entities = (kwargs.pop("agent_radius", 0.1) * 2 + 0.05)

    ''' Agent entities '''
    self.n_agents = kwargs.pop("n_agents", 6)
    self.n_teams = kwargs.pop("teams", 2)
    

    ''' Goal entities '''
    self.n_goals = kwargs.pop("n_goals", 7)
    self.goal_color = kwargs.pop("goal_colour", Color.YELLOW)

    self.flat_goal_reward = kwargs.pop("flat_goal_reward", 100)
    self.flat_goal_reward = torch.full((self.batch_dim,), self.flat_goal_reward, device=self.device)

    self.goal_range = kwargs.pop("goal_range", 0.4)
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
      batch_dim=batch_dim,  # Number of environments
      device=device,  # Use your hardware (GPU/CPU)
      substeps=3,  # Substeps for simulation accuracy
      collision_force=500,  # Collision force for agent interaction
      dt=0.1,  # Simulation timestep
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
    self.total_goals = {}

    u_range=[1]
    u_multiplier=[1]
    if self.use_influence:
      u_range=[1,1]
      u_multiplier=[1,1]
    for team in range(self.n_teams):
      self.teams[team] = []
      self.total_goals[team] = torch.zeros(batch_dim, device=device)
      for agent_num in range(int(self.n_agents)):
        agent = Agent(
          name=f"team_{team}_agent_{agent_num}",
          collide=True,
          rotatable=True,
          color=known_colors[team],
          render_action=True,
          shape=Triangle(),
          u_range=u_range,  # Ranges for actions
          u_multiplier=u_multiplier,  # Action multipliers
          dynamics=BoidDynamics(world=world, team=team, use_influence=self.use_influence)
        )
        agent.group = f"team_{team}"  # team_0 or team_1
        if team == 0:
          agent.team = torch.zeros((batch_dim,1), device=device)
        else:
          agent.team = torch.ones((batch_dim,1), device=device)

        agent.pos_rew = torch.zeros(
          batch_dim, device=device
        )  # Tensor that will hold the position reward fo the agent
        agent.agent_collision_rew = (
          agent.pos_rew.clone()
        )  # Tensor that will hold the collision reward fo the agent

        if self.use_influence:
          agent.influence = torch.zeros((batch_dim,1), device=device)

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
    # Build intrinsic node features from the agent's own state.
    if agent == self.world.agents[0]:
      # Stack the goal positions for the current batch: shape [batch_dim, n_goals, 2]
      self.goal_positions = torch.stack([goal.state.pos for goal in self.goals], dim=1)
      # Expand the agent position to shape [batch_dim, 1, 2] for broadcasting
    agent_pos = agent.state.pos.unsqueeze(1)
    
    # Compute distances using torch.cdist: returns shape [batch_dim, 1, n_goals]
    distances = torch.cdist(agent_pos, self.goal_positions).squeeze(1)  # now shape: [batch_dim, n_goals]

    # Compute the vector differences: shape [batch_dim, n_goals, 2]
    vec = self.goal_positions - agent_pos
    # Compute absolute angles (in radians) of these vectors using atan2
    angles = torch.atan2(vec[:, :, 1], vec[:, :, 0])  # shape: [batch_dim, n_goals]
    
    # Ensure the agent's heading is 1D (shape [batch_dim])
    agent_heading = agent.state.rot.squeeze(-1) if agent.state.rot.dim() > 1 else agent.state.rot
    
    # Compute the relative angles and wrap them to [-pi, pi]
    two_pi = 2 * math.pi
    rel_angles = torch.remainder(angles - agent_heading.unsqueeze(1) + math.pi, two_pi) - math.pi
    sorted_distances, indices = torch.sort(distances, dim=1)
    sorted_rel_angles = rel_angles.gather(dim=1, index=indices)

    obs = {
        "goal_distances": sorted_distances,
        "goal_rel_angles": sorted_rel_angles,
        "pos": agent.state.pos,
        "vel": agent.state.vel,
        "rot": agent.state.rot,
        "team": agent.team,
    }
    if self.use_influence:
       obs["influence"] = agent.influence
    #print([f"{key}: {obs[key].shape}, " for key in obs.keys()])
    return obs

  #############################################
  ############## reward function ##############
  #############################################

  def reward(self, agent: Agent):
      team = agent.dynamics.team

      # Negative penalty if touching boundaries.
      # (Assumes agent.state.pos has shape [batch_dim, 2].)
      coll_pen = torch.where(
          (torch.abs(agent.state.pos[:, 0]) == self.world_size_x) |
          (torch.abs(agent.state.pos[:, 1]) == self.world_size_y),
          -5, 0
      )

      # Compute team-specific rewards and capture flags once per timestep.
      # We use the first agent to do the per-timestep computation.
      if agent == self.world.agents[0]:
          # Update the goals positions for the current batch.
          self.goals_pos = torch.stack([g.state.pos for g in self.goals], dim=1)  # shape: [batch_dim, n_goals, 2]

          # Create dictionaries to store rewards and captured flags per team.
          self.team_goal_reward = {}
          self.team_captured_goals = {}
          for t in range(self.n_teams):
              team_agents = self.teams[t]  # agents for team t
              # Stack positions: shape [batch_dim, n_team_agents, 2]
              team_agents_pos = torch.stack([a.state.pos for a in team_agents], dim=1)
              # Compute distances between each team agent and every goal:
              # resulting shape: [batch_dim, n_team_agents, n_goals]
              dists = torch.cdist(team_agents_pos, self.goals_pos)
              # Count, per environment and per goal, how many team agents are within the goal range.
              # Sum along the agents dimension (dim=1) → shape: [batch_dim, n_goals]
              agents_count = torch.sum((dists < self.goal_range).int(), dim=1)
              # Determine capture: each goal is captured for team t if count >= threshold.
              captured = agents_count >= self.goal_threshold  # shape: [batch_dim, n_goals] (bool)
              self.team_captured_goals[t] = captured
              # Compute the flat reward per environment: number of captured goals × flat_goal_reward.
              # Here, captured.sum(dim=1) gives a [batch_dim] tensor.
              self.team_goal_reward[t] = captured.sum(dim=1).float() * self.flat_goal_reward
              self.total_goals[t] += captured.sum(dim=1).float()
          #print(f"team 0: {self.total_goals[0]}, team 1 {self.total_goals[1]}")

      # Handle goal respawning once per timestep.
      # We use the last agent to perform the respawn logic.
      if agent == self.world.agents[-1]:
          if self.goal_respawn:
              # Combine capture flags from all teams using a logical OR.
              # This yields a [batch_dim, n_goals] boolean tensor.
              captured_any = torch.zeros_like(self.team_captured_goals[0], dtype=torch.bool)
              for t in range(self.n_teams):
                  captured_any = captured_any | self.team_captured_goals[t]

              # For each goal index, if any environment element is captured, update that goal's position.
              # First, get the positions of all agents (for computing occupied positions).
              occupied_positions_agents = [torch.stack([a.state.pos for a in self.world.agents], dim=1)]
              for i, goal in enumerate(self.goals):
                  # captured_mask: [batch_dim] bool tensor for goal i.
                  captured_mask = captured_any[:, i]
                  if captured_mask.any():
                      # Compute occupied positions for other goals (skip goal i).
                      occupied_positions_goals = [
                          o.state.pos.unsqueeze(1)
                          for j, o in enumerate(self.goals) if j != i
                      ]
                      occupied_positions = torch.cat(occupied_positions_agents + occupied_positions_goals, dim=1)
                      # Find new positions for the captured goal.
                      pos = ScenarioUtils.find_random_pos_for_entity(
                          occupied_positions,
                          env_index=None,
                          world=self.world,
                          min_dist_between_entities=self.min_distance_between_entities,
                          x_bounds=(-self.world.x_semidim, self.world.x_semidim),
                          y_bounds=(-self.world.y_semidim, self.world.y_semidim),
                      )
                      # Update the goal's position for each environment where it was captured.
                      new_positions = pos[captured_mask].squeeze(1)
                      goal.state.pos[captured_mask] = new_positions

      # Finally, return the reward for the agent's team (plus collision penalty).
      # The reward tensor is per environment.
      team_reward = self.team_goal_reward.get(team, torch.zeros_like(self.flat_goal_reward))
      return team_reward #+ coll_pen

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



    for agent in self.world.agents:
      if agent == self.world.agents[0]:
      # Stack the goal positions for the current batch: shape [batch_dim, n_goals, 2]
        self.goal_positions = torch.stack([goal.state.pos for goal in self.goals], dim=1)
      # Expand the agent position to shape [batch_dim, 1, 2] for broadcasting
      agent_pos = agent.state.pos.unsqueeze(1)
    
    # Compute distances using torch.cdist: returns shape [batch_dim, 1, n_goals]
      distances = torch.cdist(agent_pos, self.goal_positions).squeeze(1)  # now shape: [batch_dim, n_goals]
      # if self.use_influence:
         
      #   influence_val = agent.influence[env_index].item()

      #   start = agent.state.pos[env_index]
      #   #end = agent.state.pos[env_index] + influence_val * torch.tensor([0,-1])
      #   end = agent.state.pos[env_index] + torch.tensor([0,1], device=self.world.device)
      #   arrow = make_arrow(start, end, arrow_width=0.05)
      #   geoms.append(arrow)

    # # # --- Draw arrows pointing to each goal ---
      for id, goal in enumerate(self.goals):
        start = agent.state.pos[env_index]
        end = goal.state.pos[env_index]
        distance = torch.sum(torch.square(start - end), axis=-1)
        if distance.item() < self.agent_obs_range:
          arrow = make_arrow(start, end, arrow_width=0.05)
          geoms.append(arrow)
      for other_agent in self.world.agents:
        start = agent.state.pos[env_index]
        end = other_agent.state.pos[env_index]
        distance = torch.sum(torch.square(start - end), axis=-1)
        if distance.item() < self.agent_obs_range:
          arrow = make_arrow(start, end, arrow_width=0.05)
          geoms.append(arrow)
      
    return geoms

def compute_relative_angle_and_dist(agent_heading, agent_pos, goal_pos):
    """
    Compute the relative angle (in radians) between the agent's heading and the vector pointing to the goal.
    """
    # Compute vector from agent to goal
    vec = goal_pos - agent_pos  # shape: [batch_dim, 2]
    
    # Compute angle to goal using element-wise atan2.
    # Note: We use vec[:, 1] for the y-component and vec[:, 0] for the x-component.
    angle_to_goal = torch.atan2(vec[:, 1], vec[:, 0])  # shape: [batch_dim]
    
    # Ensure agent_heading is 1D.
    if agent_heading.dim() > 1:
        agent_heading = agent_heading.squeeze(1)
    
    # Compute relative angle and wrap it to [-pi, pi].
    two_pi = 2 * math.pi
    rel_angle = torch.remainder(angle_to_goal - agent_heading + math.pi, two_pi) - math.pi
    return rel_angle

def make_arrow(start, end, arrow_width=0.05):
    """
    Create an arrow geometry from start to end.
    
    The arrow consists of a line and an arrowhead created as a filled polygon.
    """
    # Convert start and end to numpy arrays (if they are tensors or lists)
    start_np = np.array(start if isinstance(start, (list, tuple, np.ndarray)) else start.tolist())
    end_np = np.array(end if isinstance(end, (list, tuple, np.ndarray)) else end.tolist())
    
    # Create the line geometry between start and end.
    line = rendering.Line(start=tuple(start_np), end=tuple(end_np), width=1)
    
    # Compute normalized direction vector.
    direction = end_np - start_np
    length = np.linalg.norm(direction)
    if length == 0:
        return line  # Return just the line if start and end are identical.
    direction_norm = direction / length
    # Perpendicular vector (rotated by 90 degrees).
    perp = np.array([-direction_norm[1], direction_norm[0]])
    
    # Determine the arrowhead dimensions.
    head_length = arrow_width * 3  # Adjust this multiplier as needed.
    head_base = end_np - direction_norm * head_length
    left_point = head_base + perp * arrow_width
    right_point = head_base - perp * arrow_width
    # Create the arrowhead polygon using the three computed vertices.
    arrow_head = rendering.make_polygon(
        [tuple(end_np), tuple(left_point), tuple(right_point)],
        filled=True,
        draw_border=True
    )
    # Group the line and arrowhead into a compound geometry.
    arrow_geom = rendering.Compound([line, arrow_head])
    return arrow_geom

class BoidDynamics(Dynamics):
    def __init__(self, world, constant_speed=0.65, max_steering_rate=1*math.pi, team=0, use_influence=True):
        super().__init__()
        self.constant_speed = constant_speed
        self.max_steering_rate = max_steering_rate  # max radians per second
        self.world = world
        self.team = team
        self.use_influence = use_influence

    @property
    def needed_action_size(self):
        return 2 if self.use_influence else 1 

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
        if self.use_influence:
          self._agent.influence = torch.zeros((self.world.batch_dim,1), device=self.world.device)
        self._agent.state.force = torch.zeros_like(self._agent.state.force)

    def zero_grad(self):
        pass

    def clone(self):
        return BoidDynamics(self.world, constant_speed=self.constant_speed, max_steering_rate=self.max_steering_rate, team=self.team, use_influence=self.use_influence)
    
    def process_action(self):
        dt = self.world.dt
        #print(dt, self.constant_speed)
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
        if self.use_influence:
          self._agent.influence = (self._agent.influence + (self._agent.action.u[:, 1].unsqueeze(1).clamp(-1, 1) * 0.5)).clamp(0, 1)
        # Set force to zero to avoid external forces
        #self._agent.state.force = torch.zeros_like(self._agent.state.force)



class Triangle(Shape):
  def __init__(self, base: float = 0.1, height: float = 0.15):
    assert base > 0, f"Base must be > 0, got {base}"
    assert height > 0, f"Height must be > 0, got {height}"
    self.base = base
    self.height = height

  def is_point_in_triangle(self, x: float, y: float) -> bool:
    # Check if point is within the bounds of the triangle's base and height
    if y < -self.height/2 or y > self.height/2:  # Out of vertical bounds
      return False
    if x < -self.base / 2 or x > self.base / 2:  # Out of horizontal bounds
      return False

    y += (self.height/2)
    slope = self.height / (self.base / 2)
    if x < 0:  # x is to the left of the center and is negative
      max_y = slope * ((self.base/2) + x)
      return y <= max_y
    elif x > 0:  # x is to the right of the center and is positive
      max_y = self.height - slope * (x)
      return y <= max_y
    else:
      return True

  def closest_point_on_segment(self, P, A, B):
    """
    Calculate the closest point from point P to the line segment AB.

    Parameters:
      P (np.array): The point from which we are projecting.
      A (np.array): The start point of the segment.
      B (np.array): The end point of the segment.

    Returns:
      np.array: The closest point on the segment.
    """
    # Vector AB
    AB = B - A
    # Vector AP
    AP = P - A
    # Projection scalar t
    t = np.dot(AP, AB) / np.dot(AB, AB)

    # If the projection is within the segment, return the projected point
    if 0 <= t <= 1:
      closest_point = A + t * AB
    # If the projection falls before A, return A
    elif t < 0:
      closest_point = A
    # If the projection falls after B, return B
    else:
      closest_point = B

    return closest_point

  def shorten_vector_to_triangle(self, vector):
    """
    Shortens a vector so it stops at the perimeter of a triangle centered at the origin.

    Parameters:
        vector (np.array): The 2D vector originating from the origin.

    Returns:
        tuple: The shortened vector that ends at the triangle's perimeter as (x, y).
    """
    # Triangle vertices (centered at origin)
    A = np.array([-self.base / 2, -self.height / 2])  # Left base
    B = np.array([self.base / 2, -self.height / 2])   # Right base
    C = np.array([0, self.height / 2])                # Top

    # Determine which side of the triangle the vector is pointing to
    if vector[0] > 0:
      # Vector is pointing to the right, closest edge is BC
      closest_point = self.closest_point_on_segment(vector, B, C)
    else:
      # Vector is pointing to the left, closest edge is AC
      closest_point = self.closest_point_on_segment(vector, A, C)

    # Return as tuple with normal float type (not np.float64)
    return tuple(float(x) for x in closest_point)

  def get_delta_from_anchor(self, anchor):
    x, y = anchor
    x_box = x * self.base / 2
    y_box = y * self.height / 2

    if self.is_point_in_triangle(x_box, y_box):
      return float(x_box), float(y_box)  # Convert to plain float
    else:
      return self.shorten_vector_to_triangle([x_box, y_box])

  def moment_of_inertia(self, mass: float):
    return (1 / 18) * mass * (self.base**2 + self.height**2)

  def circumscribed_radius(self):
    return math.sqrt(self.base**2 + self.height**2) / 2

  def get_geometry(self) -> "Geom":
    # Vertices of the triangle (centered at the origin)
    A = (-self.height * 0.5, self.base * 0.5)  # Left base
    B = (-self.height * 0.5, -self.base * 0.5)   # Right base
    C = (self.height * 0.5, 0)                # Top

    return rendering.make_polygon([A, B, C])



class SenseSphere(Sensor):
  def __init__(self, world, range=1.0, team=0):
    super().__init__(world)
    self.range = range
    self._last_measurement = None
    self.team=team

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
      # if distance <= self.range:
      #   observations.append(torch.concat([rel_pos, rel_vel], axis=1))
      # else:
      #   observations.append(torch.zeros((1,4)))
      observations.append(torch.concat([rel_pos, rel_vel], axis=1))
    return torch.stack(observations)


  def render(self, env_index: int = 0) -> "List[Geom]":
    # if not self._render:
    #     return []

    geoms: List[rendering.Geom] = []

    # Render the range of the SenseSphere as a circle around each agent
    circle = rendering.make_circle(radius=self.range)  # Create the sensor's circle based on range
    if self.team == 0:
      circle.set_color(1, 0, 0, alpha=0.05)  # Set the color to blue with transparency
    if self.team == 0:
      circle.set_color(0, 1, 0, alpha=0.05)  # Set the color to blue with transparency
    xform = rendering.Transform()
    xform.set_translation(*self.agent.state.pos[env_index])  # Position the circle at the agent's position
    circle.add_attr(xform)

    geoms.append(circle)
    return geoms

  def to(self, device: torch.device):
    #self.range = self.range.clone().detach().requires_grad_(True)
    self.range = torch.tensor(self.range, device=device)  # Ensure range is a tensor
