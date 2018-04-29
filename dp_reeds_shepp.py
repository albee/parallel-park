import math
import numpy
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from time import sleep
import copy
import bisect
from collections import Counter
np = numpy
from car import *
from obstacle import *

### Mesh ###
class Mesh:
	"""
	A dynamic programming mesh, configured to solve parallel parking
	"""
	def __init__(self, car, obstacles, x_size, y_size, theta_size, phi_size, x_max, y_max, phi_max, X0, K_max):
		"""
		Creates a state space grid and initializes X, U
		""" 
		self.obstacles = obstacles
		self.car = car

		self.K_max = K_max
		self.X0 = X0 #goal state (J* = 0), specified as mesh coordinates
		self.X_size = np.matrix([[x_size], [y_size], [theta_size]])
		self.phi_size = phi_size
		self.phi_max = phi_max

		# discretized state: [x; y; theta]
		self.x_disc = np.linspace(0, x_max, num=x_size)
		self.y_disc = np.linspace(0, y_max, num=y_size)
		self.theta_disc = np.linspace(0, 2*math.pi - 2*math.pi/theta_size,  num=theta_size)
		# print(self.x_disc)
		# print(self.y_disc)
		# print(self.theta_disc)

		# discretized input: [u; phi]
		self.v_disc = np.array([-float(x_max)/(x_size-1), float(x_max)/(x_size-1)]) #discretized to size of mesh spacing
		self.phi_disc = np.linspace(-phi_max, phi_max, num=phi_size)

		# adjust car value
		self.car.x = self.x_disc[self.X0[0,0]]
		self.car.y = self.y_disc[self.X0[1,0]]
		self.car.theta = self.theta_disc[self.X0[2,0]]

		"""
		mesh: (x_size X y_size X theta_size X 3)
		Optimal cost-to-go, optimal control at each grid point. mesh has 3x1 opt = [J*;v*;phi*] stored at each point, mesh[#x,#y,#theta,:]
		"""
		self.mesh = self.init_mesh(0)

	def init_mesh(self, X0_J):
		"""
		Creates the initial mesh, and specifies a J* and U* at each mesh point.
		Intially, each J* and U* is inf, except for the goal state, which is set to J*=0
		XK_J = J* for goal state, h()
		"""
		X_size = self.X_size
		opt = np.asmatrix(np.full(3,1))
		mesh = np.full((X_size[0], X_size[1], X_size[2], 3), np.inf)
		mesh[self.X0[0], self.X0[1], self.X0[2], 0] = X0_J # set cost-to-go for goal state
		return mesh

	def collision_check(self):
		"""
		Determines if the given state, agent boundaries, and obstacles, cause a collision
		"""
		p_WORLD = self.car.CAR_to_WORLD()
		for obstacle in self.obstacles:
			if obstacle.is_colliding(p_WORLD):
				return True
		return False

	def dp(self):
		"""
		Solves the given problem using dynamic programming
		"""
		# March forward for K_max iterations
		for k in range(self.K_max):
			print(k)
			# Iterate over the grid
			for m in range(self.X_size[0,0]): # x iter
				for n in range(self.X_size[1,0]): # y iter
					for p in range(self.X_size[2,0]): # theta iter
						cost_min = np.inf
						if (m == self.X0[0] and n == self.X0[1] and p == self.X0[2]): # don't change cost if is goal state
							cost_min = 0
						cost_qr = np.inf
						v_min = np.inf
						phi_min = np.inf

						# Iterate over discretized input
						for q in range(2): # v iter
							for r in range(self.phi_size): # phi iter
								# Find where this control takes you at k+1
								x_k1, y_k1, theta_k1 = self.f(self.x_disc[m], self.y_disc[n], self.theta_disc[p], self.v_disc[q], self.phi_disc[r])
							
								# Interpolate the cost-to-go at where you end up and find the total cost-to-go for J_k+1
								cost_qr = self.find_J_k1(self.x_disc[m], self.y_disc[n], self.theta_disc[p], x_k1, y_k1, theta_k1)
								# if (m==5 and n ==5 and p == 2):
								# 	print(cost_qr)
								# If lower than other input cost, set new J*_k+1 for the grid point
								if cost_qr < cost_min:
									cost_min = cost_qr
									v_min = self.v_disc[q]
									phi_min = self.phi_disc[r]
						if (cost_min < self.mesh[m,n,p,0]):
							self.mesh[m,n,p,0] = cost_min
							self.mesh[m,n,p,1] = v_min
							self.mesh[m,n,p,2] = phi_min

	def f(self, x, y, theta, v, phi):
		"""
		State dynamics, finds state k+1 given x_k and u_k
		"""
		x_k1 = x + v*math.cos(theta)
		y_k1 = y + v*math.sin(theta)
		theta_k1 = (theta + v/(self.car.l) * math.tan(phi)) % (math.pi*2) # note: car length set to 1 for convenience

		return x_k1, y_k1, theta_k1

	def find_J_k1(self, x_k, y_k, theta_k, x_k1, y_k1, theta_k1):
		"""
		Calculate the new J* from a chosen iteration
		"""
		# Cost at endpoint (interpolated if not a grid point)
		J_k1, v_flip = self.interp_J_k(x_k, y_k, theta_k, x_k1, y_k1, theta_k1)
		if (J_k1 == np.inf): # stop if not a valid end condition
			return J_k1

		# Cost of movement
		C_k = self.cost_to_move(x_k, y_k, theta_k, x_k1, y_k1, theta_k1, v_flip)
		J_k1 = J_k1+C_k

		return J_k1

	def interp_J_k(self, x_k, y_k, theta_k, x_k1, y_k1, theta_k1):
		"""
		Interpolate J_k from grid points near where the control iteration ended
		"""
		th_s = self.X_size[2,0]
		v_flip = False
		J_k1 = np.inf
		idx, idy, idth = self.get_neighbors(x_k1, y_k1, theta_k1)
		J = self.filter_edge_cases(idx, idy, idth, 0) #get valid neighbors for cost-to-go

		# If out of bounds, neglect
		# if (idx >= self.X_size[0,0] or idx == 0 or
		# 	idy >= self.X_size[1,0] or idy == 0):
		# 	return J_k1, v_flip

		# If colliding, neglect
		# TODO

		# Otherwise, interpolate grid values
		# Averaging interpolation (inaccurate)
		# J000 = self.mesh[idx-1, idy-1, (idth-1)%th_s, 0]
		# J001 = self.mesh[idx-1, idy-1, (idth)%th_s, 0]
		# J010 = self.mesh[idx-1, idy, (idth-1)%th_s, 0]
		# J011 = self.mesh[idx-1, idy, (idth)%th_s, 0]
		# J100 = self.mesh[idx, idy-1, (idth-1)%th_s, 0]
		# J101 = self.mesh[idx, idy-1, (idth)%th_s, 0]
		# J110 = self.mesh[idx, idy, (idth-1)%th_s, 0]
		# J111 = self.mesh[idx, idy, (idth)%th_s, 0]

		# J = [J000, J001, J010, J011, J100, J101, J110, J111]

		J2 = [i for i in J if i != np.inf]
		if (len(J2) != 0):
			J_k1 = np.mean(J2)

		# If v flipped sign, take note
		if (v_flip):
			v_flip = True

		return J_k1, v_flip

	def interp_u(self, x_k1, y_k1, theta_k1):
		"""
		Interpolate the optimal control at a non gridpoint
		"""
		idx, idy, idth = self.get_neighbors(x_k1, y_k1, theta_k1)
		# if (idx >= self.X_size[0,0] or idx == 0 or
		# 	idy >= self.X_size[1,0] or idy == 0):
		
		# v control
		v = self.filter_edge_cases(idx, idy, idth, 1) #get valid neighbors for control v
		c = Counter(v)
		most = c.most_common()[0][0] # select most common velocity
		v_star = most

		# phi control
		p = self.filter_edge_cases(idx, idy, idth, 2) #get valid neighbors for control phi
		p2 = [i for i in p if i != np.inf]
		if (len(p2) != 0):
			p_star = np.mean(p2)
		return v_star, p_star

	def get_neighbors(self, x_k1, y_k1, theta_k1):
		"""
		Get points in the 8-cube surrounding the X_k1 point
		"""
		idx = bisect.bisect(self.x_disc, x_k1) # this index and one below it are neighbors
		idy = bisect.bisect(self.y_disc, y_k1)
		idth = bisect.bisect(self.theta_disc, theta_k1)
		return idx, idy, idth

	def filter_edge_cases(self, idx, idy, idth, var):
		"""
		Filters out edge cases and returns the valid nearest neighbors
		"""
		# print(idx, idy, idth)
		th_s = self.X_size[2,0]
		if ((idx == 0) and (idy == 0)): # bottom left
			v110 = self.mesh[idx, idy, (idth-1)%th_s, var]
			v111 = self.mesh[idx, idy, (idth)%th_s, var]
			v = [v110, v111]
		elif (idx >= self.X_size[0,0] and (idy == 0)): # bottom right
			v010 = self.mesh[idx-1, idy, (idth-1)%th_s, var]
			v011 = self.mesh[idx-1, idy, (idth)%th_s, var]
			v = [v010, v011]
		elif (idy >= self.X_size[1,0] and (idx == 0)): # top left
			v100 = self.mesh[idx, idy-1, (idth-1)%th_s, var]
			v101 = self.mesh[idx, idy-1, (idth)%th_s, var]
			v = [v100, v101]
		elif (idx >= self.X_size[0,0] and idy >= self.X_size[1,0]): # top right
			v000 = self.mesh[idx-1, idy-1, (idth-1)%th_s, var]
			v001 = self.mesh[idx-1, idy-1, (idth)%th_s, var]
			v = [v000, v001]
		elif (idx == 0): # left
			v100 = self.mesh[idx, idy-1, (idth-1)%th_s, var]
			v101 = self.mesh[idx, idy-1, (idth)%th_s, var]
			v110 = self.mesh[idx, idy, (idth-1)%th_s, var]
			v111 = self.mesh[idx, idy, (idth)%th_s, var]
			v = [v100, v101, v110, v111]
		elif (idy == 0): # bottom
			v010 = self.mesh[idx-1, idy, (idth-1)%th_s, var]
			v011 = self.mesh[idx-1, idy, (idth)%th_s, var]
			v110 = self.mesh[idx, idy, (idth-1)%th_s, var]
			v111 = self.mesh[idx, idy, (idth)%th_s, var]
			v = [v010, v011, v110, v111]
		elif (idx >= self.X_size[0,0]): # right
			v000 = self.mesh[idx-1, idy-1, (idth-1)%th_s, var]
			v001 = self.mesh[idx-1, idy-1, (idth)%th_s, var]
			v010 = self.mesh[idx-1, idy, (idth-1)%th_s, var]
			v011 = self.mesh[idx-1, idy, (idth)%th_s, var]
			v =[v000, v001, v010, v011]
		elif (idy >= self.X_size[1,0]): # top
			v000 = self.mesh[idx-1, idy-1, (idth-1)%th_s, var]
			v001 = self.mesh[idx-1, idy-1, (idth)%th_s, var]
			v101 = self.mesh[idx, idy-1, (idth)%th_s, var]
			v100 = self.mesh[idx, idy-1, (idth-1)%th_s, var]
			v = [v000, v001, v101, v100]
		else: # normal interp
			v000 = self.mesh[idx-1, idy-1, (idth-1)%th_s, var] # v_x_y_theta
			v001 = self.mesh[idx-1, idy-1, (idth)%th_s, var]
			v010 = self.mesh[idx-1, idy, (idth-1)%th_s, var]
			v011 = self.mesh[idx-1, idy, (idth)%th_s, var]
			v100 = self.mesh[idx, idy-1, (idth-1)%th_s, var]
			v101 = self.mesh[idx, idy-1, (idth)%th_s, var]
			v110 = self.mesh[idx, idy, (idth-1)%th_s, var]
			v111 = self.mesh[idx, idy, (idth)%th_s, var]
			v = [v000, v001, v010, v011, v100, v101, v110, v111]
		return v

	def cost_to_move(self, x_k, y_k, theta_k, x_k1, y_k1, theta_k1, v_flip):
		"""
		Calculate the additive cost of performing a control iteration
		"""
		dist = np.linalg.norm(np.array((x_k, y_k)) - np.array((x_k1, y_k1)))
		if (v_flip): 
			g_k1 = dist + 1 # change in velocity command
		else:
			g_k1 = dist
		return g_k1

	def find_u_opt(self, x_start, y_start, theta_start):
		"""
		Returns the optimal control and state trajectory based on the computed dp mesh
		"""
		iter = 0
		v_star = self.mesh[x_start, y_start, theta_start, 1]
		phi_star = self.mesh[x_start, y_start, theta_start, 1]
		x_k = x_start
		y_k = y_start
		th_k = theta_start

		# State history for plotting
		x_k_hist = [x_k]
		y_k_hist = [y_k]
		th_k_hist = [th_k]

		if (v_star == np.inf or phi_star == np.inf):
			print('mesh error!')
			return 'blargh'

		while (np.linalg.norm(np.array((x_k, y_k)) - np.array((self.x_disc[self.X0[0,0]]), self.y_disc[self.X0[1,0]])) > (self.x_disc[1]-self.x_disc[0])/2 ):
			iter += 1
			if (iter > 500):
				break
			#move
			x_k1, y_k1, th_k1 = self.f(x_k, y_k, th_k, v_star, phi_star)

			#interpolate to get u*
			v_star, phi_star = self.interp_u(x_k1, y_k1, th_k1)

			x_k = x_k1
			y_k = y_k1
			th_k = th_k1
			x_k_hist.append(x_k)
			y_k_hist.append(y_k)
			th_k_hist.append(th_k)
		return x_k_hist, y_k_hist, th_k_hist

	def plot_mesh(self):
		"""
		Plots the x, y, grid points of the mesh
		"""
		for x in self.x_disc:
			for y in self.y_disc:	
				plt.scatter(x, y, c='black')

	def plot_J(self):
		"""
		Plots the J for the mesh
		"""
		xid  = range(self.X_size[0,0])
		yid  = range(self.X_size[1,0])
		thid = range(self.X_size[2,0])
		xpts = []
		ypts = []
		thpts = []
		Jpts = []
		for i in xid:
			for j in yid:
				for k in thid:
					xpts.append(self.x_disc[i])
					ypts.append(self.y_disc[j])
					thpts.append(self.theta_disc[k])
					Jpts.append(self.mesh[i,j,k,0])
		ax2.scatter(xpts, ypts, thpts, zdir='z', c=Jpts, s=100)

	def plot_goal_state(self):
		"""
		Plots the car state of the goal
		"""
		# adjust car value
		plot_car = copy.deepcopy(self.car)
		plot_car.x = self.x_disc[self.X0[0,0]]
		plot_car.y = self.y_disc[self.X0[1,0]]
		plot_car.theta = self.theta_disc[self.X0[2,0]]
		plot_car.plot_car()

	#---Animate line trajectory of x, y------------------------------------------------
	def init_anim(self):
		line.set_data([], [])
		return line,

	def update_anim(self, frame):
		xdata.append(x_k_hist[frame])
		ydata.append(y_k_hist[frame])
		line.set_data(xdata, ydata)
		return line,

	def show_anim(self, fig, x_k_hist, y_k_hist, th_k_hist):
		anim = animation.FuncAnimation(fig, self.update_anim, frames=len(x_k_hist),
                              interval=500, blit=True, init_func=self.init_anim, repeat=False)
		plt.show()
	#---Animate line trajectory of x, y------------------------------------------------

	#---Animate car parking of x, y, theta---------------------------------------------
	def init_anim_car(self):
		return line12, line13, line34, line24, center

	def update_anim_car(self, frame):
		car.set_car(x_k_hist[frame], y_k_hist[frame], th_k_hist[frame])

		pts = car.CAR_to_WORLD() # x1, y1, x2, y2, x3, y3, x4, y4
		print(pts)
		x12data, y12data = [pts[0], pts[2]], [pts[1], pts[3]]
		x13data, y13data = [pts[0], pts[4]], [pts[1], pts[5]]
		x34data, y34data = [pts[4], pts[6]], [pts[5], pts[7]]
		x24data, y24data = [pts[2], pts[6]], [pts[3], pts[7]]

		center.set_data(x_k_hist[frame], y_k_hist[frame])
		line12.set_data(x12data, y12data)
		line13.set_data(x13data, y13data)
		line34.set_data(x34data, y34data)
		line24.set_data(x24data, y24data)
		return line12, line13, line34, line24, center

	def show_anim_car(self, fig, x_k_hist, y_k_hist, th_k_hist):
		anim = animation.FuncAnimation(fig, self.update_anim_car, frames=len(x_k_hist),
                              interval=250, blit=True, init_func=self.init_anim_car)
		anim.save('anim1.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
		plt.show()
	#---Animate car parking of x, y, theta---------------------------------------------


def init_obstacles(scenario):
	"""
	Creates obstacles in the grid for collision checking; rectangular for now
	"""
	obstacles = []
	if (scenario == 0):
		obst1 = Obstacle(0,5,0,2.5) #xmin, xmax, ymin, ymax
		obst2 = Obstacle(10,15,0,2.5)
		obstacles.append(obst1)
		obstacles.append(obst2)
	return obstacles

def init_car(scenario):
	"""
	Creates a car agent
	"""
	if (scenario == 0):
		# Car
		x0 = 5
		y0 = 5
		theta0 = 0
		x1, y1 = -.2, .5
		x2, y2 =  x1, -y1
		x3, y3 = 1.2,  y1
		x4, y4 =  x3,  -y1

		car = Car(x0, y0, theta0, x1, y1, x2, y2, x3, y3, x4, y4)
	return car

def init_mesh(scenario, car, obstacles):
	"""
	Creates a mesh instance
	"""
	if (scenario == 0):
		x_size = 21
		y_size = 21
		theta_size = 15
		phi_size = 8
		phi_max = np.pi/3
		x_max = 15
		y_max = 5
		K_max = 10 # max iterations
		X0 = np.matrix([1,1,0]).T # value iteration goal state

	if (scenario == 1):
		x_size = 15
		y_size = 5
		theta_size = 8
		phi_size = 8
		phi_max = np.pi/3
		x_max = 15
		y_max = 5
		K_max = 10 # max iterations
		X0 = np.matrix([2,2,0]).T # value iteration goal state

	if (scenario == 2):
		x_size = 45
		y_size = 15
		theta_size = 36
		phi_size = 20
		phi_max = np.pi/3
		x_max = 15
		y_max = 5
		K_max = 10 # max iterations
		X0 = np.matrix([4,4,9]).T # value iteration goal state

	mesh = Mesh(car, obstacles, x_size, y_size, theta_size, phi_size, x_max, y_max, phi_max, X0, K_max)
	return mesh, x_max, y_max

if __name__ == '__main__':
	#Create a Reeds-Shepp instance


	obstacles =          init_obstacles(0)
	car =                init_car(0)
	mesh, x_max, y_max = init_mesh(1, car, obstacles)	

	#---------------------------------------------
	# Init 2D Plot
	fig, ax = plt.subplots()
	ax.set_aspect(1)
	ax.set_xlim([0,x_max])
	ax.set_ylim([0,y_max])

	line, = plt.plot([], [], lw=2, animated=True) # takes the one Line2D returned (empty)
	xdata, ydata = [], []

	center, = plt.plot([], [], 'bo', lw=2, animated=True, ms=6) # takes the one Line2D returned (empty)
	line12, = plt.plot([], [], lw=2, animated=True, c='b') # takes the one Line2D returned (empty)
	line13, = plt.plot([], [], lw=2, animated=True, c='b') # takes the one Line2D returned (empty)
	line34, = plt.plot([], [], lw=2, animated=True, c='b') # takes the one Line2D returned (empty)
	line24, = plt.plot([], [], lw=2, animated=True, c='b') # takes the one Line2D returned (empty)
	centerdata = []
	x12data, y12data = [], []
	x13data, y13data = [], []
	x34data, y34data = [], []
	x24data, y24data = [], []

	# Update plots
	car.plot_car()
	# for obstacle in obstacles:
	# 	obstacle.plot_obstacle(fig)
	mesh.plot_mesh()
	mesh.plot_goal_state()
	# plt.close()		

	# Init 3D Plot
	fig2, ax2 = plt.subplots(subplot_kw=dict(projection='3d'))
	ax2.set_xlabel('x, m')
	ax2.set_ylabel('y, m')
	ax2.set_zlabel('theta, rad')
	# Activate the first figure
	plt.close()
	#---------------------------------------------

	mesh.dp()
	mesh.plot_J()
	# print(mesh.mesh[6,6,0,:])

	# Plot u_opt history
	x_k_hist, y_k_hist, th_k_hist = mesh.find_u_opt(11,3,3)	
	mesh.show_anim_car(fig, x_k_hist, y_k_hist, th_k_hist)

	plt.show()
