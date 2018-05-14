import math
import matplotlib
import matplotlib.pyplot as plt
import numpy
np = numpy

### CAR ###
class Car:
	def __init__(self, x, y, th, x1, y1, x2, y2, x3, y3, x4, y4):
		#car corners, CAR frame
		self.x = x
		self.y = y 
		self.th = th 
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.x3 = x3
		self.y3 = y3
		self.x4 = x4
		self.y4 = y4
		self.l = 1

	def set_car(self, x_new, y_new, th_new):
		"""
		Moves the car to a specified state
		"""
		self.x = x_new
		self.y = y_new
		self.th = th_new

	def CAR_to_WORLD(self):
		"""
		Returns car variables in WORLD frame
		"""
		R_C_W = self.R_C_W()

		p1 = R_C_W*np.matrix([[self.x1],[self.y1]]) + np.matrix([[self.x],[self.y]])
		p2 = R_C_W*np.matrix([[self.x2],[self.y2]]) + np.matrix([[self.x],[self.y]])
		p3 = R_C_W*np.matrix([[self.x3],[self.y3]]) + np.matrix([[self.x],[self.y]])
		p4 = R_C_W*np.matrix([[self.x4],[self.y4]]) + np.matrix([[self.x],[self.y]])
		x1 = p1[0,0]
		y1 = p1[1,0]
		x2 = p2[0,0]
		y2 = p2[1,0]
		x3 = p3[0,0]
		y3 = p3[1,0]
		x4 = p4[0,0]
		y4 = p4[1,0]
		return [x1, y1, x2, y2, x3, y3, x4, y4]

	def R_C_W(self):
		"""
		Rotation matrix
		"""
		th = self.th
		R_C_W = np.matrix([[math.cos(th), -math.sin(th)],
			 		       [math.sin(th), math.cos(th)]])
		return R_C_W

	def plot_car(self):
		"""
		Plots the car at its current position
		"""
		p_WORLD = self.CAR_to_WORLD()
		plt.scatter([p_WORLD[0], p_WORLD[2], p_WORLD[4], p_WORLD[6]], [p_WORLD[1], p_WORLD[3], p_WORLD[5], p_WORLD[7]], c='g') #corners
		plt.plot([p_WORLD[0], p_WORLD[2], p_WORLD[6], p_WORLD[4], p_WORLD[0]], [p_WORLD[1], p_WORLD[3], p_WORLD[7], p_WORLD[5], p_WORLD[1]], c='g') #rectangle
		plt.scatter(self.x, self.y, c='g') #center