# !pip install vmas
from vmas.simulator.core import Shape
import numpy as np
import math

class Triangle(Shape):
  def __init__(self, base: float = 10, height: float = 10):
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
    from vmas.simulator import rendering

    # Vertices of the triangle (centered at the origin)
    A = (-self.base / 2, -self.height / 2)  # Left base
    B = (self.base / 2, -self.height / 2)   # Right base
    C = (0, self.height / 2)                # Top

    # Return geometry as a polygon
    return rendering.make_polygon([A, B, C])
