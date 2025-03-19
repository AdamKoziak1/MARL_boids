class BoidAgent:
  def __init__(self, collide, color, render_action, position, velocity, sensor_range, max_speed, max_force):
    super().__init__()

    self.collide = collide  # Whether the agent collides with others
    self.color = color  # Color of the agent for rendering
    self.render_action = render_action  # Whether to render actions
    self.position = position  # Initial position
    self.velocity = velocity  # Initial velocity
    self.sensor_range = sensor_range  # Observation range
    self.max_speed = max_speed  # Maximum speed of the agent
    self.max_force = max_force  # Maximum force that can be applied

    # VMAS-specific attributes
    self.pos_rew = None  # Placeholder for position reward
    self.agent_collision_rew = None  # Placeholder for collision reward

  def update(self, steering_force):
    """Update the boid's velocity and position based on the steering force."""
    self.velocity += steering_force
    speed = self.velocity.norm()
    if speed > self.max_speed:
        self.velocity = self.velocity / speed * self.max_speed
    self.position += self.velocity

  def apply_force(self, force):
    """Apply a force to the boid."""
    self.velocity += force
