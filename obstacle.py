import math
import matplotlib
import matplotlib.pyplot as plt
import numpy
np = numpy

### OBSTACLE ###
class Obstacle:
	def __init__(self, x_min, x_max, y_min, y_max):
		self.x_min = x_min
		self.x_max = x_max
		self.y_min = y_min
		self.y_max = y_max

	def is_colliding(self, p_WORLD):
		"""
		Determines if a 4-gon is colliding with this obstacle. Assumes non-rotated rectangles
		"""
		x1 = p_WORLD[0]
		y1 = p_WORLD[1]
		x2 = p_WORLD[2]
		y2 = p_WORLD[3]
		x3 = p_WORLD[4]
		y3 = p_WORLD[5]
		x4 = p_WORLD[6]
		y4 = p_WORLD[7]
		#Check obstacle half planes (broad phase)
		if ( all(x > self.x_max for x in [x1, x2, x3, x4]) or 
			 all(x < self.x_min for x in [x1, x2, x3, x4]) or
			 all(y > self.y_max for y in [y1, y2, y3, y4]) or
			 all(y < self.y_min for y in [y1, y2, y3, y4]) ):
			return False

		#Check agent half planes (narrrow phase)
		# float sign (fPoint p1, fPoint p2, fPoint p3)
		# {
		#     return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
		# }

		# bool PointInTriangle (fPoint pt, fPoint v1, fPoint v2, fPoint v3)
		# {
		#     bool b1, b2, b3;

		#     b1 = sign(pt, v1, v2) < 0.0f;
		#     b2 = sign(pt, v2, v3) < 0.0f;
		#     b3 = sign(pt, v3, v1) < 0.0f;

		#     return ((b1 == b2) && (b2 == b3));
		# }
		return True

	def calc_sign_d(xa, xb, xc, ya, yb, yc):
		"""
		Finds whether the point (xc, yc) is above or below the line formed by (xa, ya)
		and (xb, yb)
		"""

	def plot_obstacle(self, fig):
		"""
		Plots the obstacle
		"""
		plt.figure(fig.number)
		plt.plot([self.x_min, self.x_max, self.x_max, self.x_min, self.x_min], [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min], 'ro-') #rectangle

