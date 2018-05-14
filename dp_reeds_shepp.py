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
from scipy.interpolate import RegularGridInterpolator
np = numpy
from car import *
from obstacle import *

### Mesh ###
class Mesh:
	"""
	A dynamic programming mesh, configured to solve parallel parking
	"""
	def __init__(self, car, obstacles, x_len, y_len, th_len, phi_len, x_max, y_max, phi_max, X0, K_max):
		"""
		Creates a state space grid and initializes X, U
		""" 
		self.obstacles = obstacles
		self.car = car
		self.goal_car = copy.deepcopy(car)

		self.K_max = K_max
		self.X0 = X0 #goal state (J* = 0), specified as mesh coordinates

		# length of discretized state
		self.x_len = x_len
		self.y_len = y_len
		self.th_len = th_len
		# discretized state: [x; y; theta]
		self.x_disc = np.linspace(0, x_max, num=x_len)
		self.y_disc = np.linspace(0, y_max, num=y_len)
		self.th_disc = np.linspace(0, 2*math.pi - 2*math.pi/th_len,  num=th_len)

		# length of discretized input
		self.phi_len = phi_len
		self.phi_max = phi_max
		# discretized input: [u; phi]
		self.v_disc = np.array([-float(x_max)/(x_len-1), float(x_max)/(x_len-1)]) #discretized to size of mesh spacing, 2 values
		self.phi_disc = np.linspace(-phi_max, phi_max, num=phi_len)

		# adjust car value
		self.car.x = self.x_disc[self.X0[0,0]]
		self.car.y = self.y_disc[self.X0[1,0]]
		self.car.th = self.th_disc[self.X0[2,0]]

		"""
		mesh: (x_size X y_size X theta_size X 3)
		Optimal cost-to-go, optimal control at each grid point. mesh has 3x1 opt = [J*;v*;phi*] stored at each point, mesh[#x,#y,#theta,:]
		"""
		self.mesh = self.init_mesh(0)
		self.mark_grid_collision()
		self.c_mesh = self.init_collision_mesh()

	def init_mesh(self, X0_J):
		"""
		Creates the initial mesh, and specifies a J* and U* at each mesh point.
		Intially, each J* and U* is inf, except for the goal state, which is set to J*=0
		XK_J = J* for goal state, h()
		"""
		mesh = np.full((self.x_len, self.y_len, self.th_len, 4), 9999)
		mesh[self.X0[0], self.X0[1], self.X0[2], 0] = X0_J # set cost-to-go for goal state
		return mesh

	def init_collision_mesh(self):
		"""
		Marks all collision-inducing inputs, as well as inputs that go out-of-bounds
		"""
		c_mesh = np.full((self.x_len, self.y_len, self.th_len, 2, self.phi_len), 0)
		# Iterate of discretized state space
		for m in range(self.x_len): # x iter
			for n in range(self.y_len): # y iter
				for p in range(self.th_len): # theta iter
					if (m == self.X0[0] and n == self.X0[1] and p == self.X0[2]): # don't change cost if is goal state
						break
					# Iterate over discretized input
					for q in range(2): # v iter
						for r in range(self.phi_len): # phi iter
							x_k1, y_k1, th_k1 = self.f(self.x_disc[m], self.y_disc[n], self.th_disc[p], self.v_disc[q], self.phi_disc[r])
							if (x_k1 > self.x_disc[self.x_len-1] or x_k1 < 0 or
								y_k1 > self.y_disc[self.y_len-1] or y_k1 < 0):
								c_mesh[m,n,p,q,r] = 1 # mark as an invalid input
							self.car.set_car(x_k1, y_k1, th_k1)
							if self.collision_check():
								c_mesh[m,n,p,q,r] = 1 # mark as an invalid input
		print('Total input checks per mesh:')
		print(c_mesh.size)
		print('Out-of-bounds input checks:')
		print(int(np.sum(c_mesh)))
		return c_mesh

	def mark_grid_collision(self):
		"""
		Deactivates grid points that have an obstacle collision
		"""
		for m in range(self.x_len): # x iter
			for n in range(self.y_len): # y iter
				for p in range(self.th_len): # th iter
					self.mesh[m,n,p,3] = 0
					for obstacle in self.obstacles:
						if obstacle.contains_point(self.x_disc[m], self.y_disc[n]):
						# Lies within an obstacle
							self.mesh[m,n,p,3] = 1
						else:
						# Full collision check
							self.car.set_car(self.x_disc[m], self.y_disc[n], self.th_disc[p])
							if self.collision_check():
								self.mesh[m,n,p,3] = 1

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
			# Reinitialize the interpolater
			RGI = RegularGridInterpolator(points=[self.x_disc, self.y_disc, self.th_disc], values=self.mesh[:,:,:,0],  bounds_error=True)
			# Iterate over the grid
			for m in range(self.x_len): # x iter
				for n in range(self.y_len): # y iter
					for p in range(self.th_len): # theta iter
						cost_min = np.inf
						if (m == self.X0[0] and n == self.X0[1] and p == self.X0[2]): # don't change cost if is goal state
							break
						if (self.mesh[m,n,p,3] == 1): # don't change cost if an obstacle point
							break
						cost_qr = np.inf
						v_min = np.inf
						phi_min = np.inf
						# Iterate over discretized input
						for q in range(2): # v iter
							for r in range(self.phi_len): # phi iter
								if (self.c_mesh[m,n,p,q,r] == 1): # skip if input causes collision
									continue
								# Find where this control takes you at k+1
								x_k1, y_k1, th_k1 = self.f(self.x_disc[m], self.y_disc[n], self.th_disc[p], self.v_disc[q], self.phi_disc[r])
							
								# Interpolate the cost-to-go at where you end up and find the total cost-to-go for J_k+1
								cost_qr = self.find_J_k1(m, n, p, x_k1, y_k1, th_k1, RGI, k, self.v_disc[q])

								# If lower than other input cost, set new J*_k+1 for the grid point
								if cost_qr < cost_min:
									cost_min = cost_qr
									v_min = self.v_disc[q]
									phi_min = self.phi_disc[r]
						if (cost_min < self.mesh[m,n,p,0]):
							self.mesh[m,n,p,0] = cost_min
							self.mesh[m,n,p,1] = v_min
							self.mesh[m,n,p,2] = phi_min

	def f(self, x, y, th, v, phi):
		"""
		State dynamics, finds state k+1 given x_k and u_k
		"""
		x_k1 = x + v*math.cos(th)
		y_k1 = y + v*math.sin(th)
		th_k1 = (th + v/(self.car.l) * math.tan(phi)) % (math.pi*2) # note: car length set to 1 for convenience

		return x_k1, y_k1, th_k1

	def find_J_k1(self, x_ind_k, y_ind_k, th_ind_k, x_k1, y_k1, th_k1, RGI, k, v_k):
		"""
		Calculate the new J* from a chosen iteration
		"""
		x_k = self.x_disc[x_ind_k]
		y_k = self.y_disc[y_ind_k]
		th_k = self.th_disc[th_ind_k]

		# Cost at endpoint (interpolated if not a grid point), also includes penalty for direction change
		J_k1 = self.interp_J_k(x_ind_k, y_ind_k, th_ind_k, x_k1, y_k1, th_k1, RGI, k, v_k)
		if (J_k1 == 9999): # stop if not a valid end condition
			return J_k1	
	
		# Cost of movement
		C_k = self.cost_to_move(x_k, y_k, th_k, x_k1, y_k1, th_k1)
		J_k1 = J_k1+C_k

		return J_k1

	def interp_J_k(self, x_ind_k, y_ind_k, th_ind_k, x_k1, y_k1, th_k1, RGI, k, v_k):
		"""
		Interpolate J_k from grid points near where the control iteration ended.
		Tests increase in order of expense.
		"""
		x_k = self.x_disc[x_ind_k]
		y_k = self.y_disc[y_ind_k]
		th_k = self.th_disc[th_ind_k]

		J_k1 = 0 
	
		idx, idy, idth = self.get_neighbors(x_k1, y_k1, th_k1) # get valid neighbors for cost-to-go
		J_cube = self.filter_edge_cases(idx, idy, idth, 0) # modify the J values

		# Catch case of all equal values (e.g. all 9999)
		if (J_cube[1:] == J_cube[:-1]):
			J_k1 = J_cube[0]
			return J_k1

		#add on if sign change	
		# print(self.mesh[:,:,:,1])
		# print('debug')
		# print(x_k1, y_k1, th_k1)
		# print(idx, idy, idth)
		# print(self.tri_interp_control(x_k1, y_k1, th_k1, idx, idy, idth, 1))
		# print(np.sign(self.tri_interp_control(x_k1, y_k1, th_k1, idx, idy, idth, 1)))
		# print(v_k)
		# v_sign = np.sign(self.tri_interp_control(x_k1, y_k1, th_k1, idx, idy, idth, 1))
		# if (v_sign != np.sign(v_k)):
		# 	J_k1 += (self.x_disc[1] - self.x_disc[0])*50 # add on equivalent of fifty grid travel

		# Go to neareast after a few iterations
		# if (k > 0):
		# 	J_cube2 = [i for i in J_cube if i < 500]
		# 	if len(J_cube2) > 2:
		# 		J_k1 = self.get_nearest_J(x_k1, y_k1, th_k1, idx, idy, idth)
		# 		return J_k1

		# If on a boundary, use the filter approximation
		if (idx >= self.x_len or idx == 0 or
			idy >= self.y_len or idy == 0):
			J_k1 += np.mean(J_cube)
			return J_k1

		# Trilinear, but initialize everything to a high value
		else:
			# print('tri')
			# print(idx, idy, idth)
			# print(x_k1, y_k1, th_k1)
			# print('tri dims')
			# print(low_th, high_th)
			# print(self.mesh[idx-1:idx+1,idy-1:idy+1,idth-1:idth+1,0].shape)
			J_k1 += self.tri_interp(x_k1, y_k1, th_k1, idx, idy, idth, 0)
		return J_k1

	def get_nearest_J(self, x_k1, y_k1, th_k1, idx, idy, idth):
		"""
		Return the Euclidean nearest grid point
		"""
		th_s = self.th_len
		if ((idx == 0) and (idy == 0)): # bottom left
			idx+=1
			idy+=1
		elif (idx >= self.x_len and (idy == 0)): # bottom right
			idx+=-1
			idy+=1
		elif (idy >= self.y_len and (idx == 0)): # top left
			idx+=1
			idy+=-1
		elif (idx >= self.x_len and idy >= self.y_len): # top right
			idx+=-1
			idy+=-1
		elif (idx == 0): # left
			idx+=1
		elif (idy == 0): # bottom
			idy+=1
		elif (idx >= self.x_len): # right
			idx+=-1
		elif (idy >= self.y_len): # top
			idy+=-1

		low_p = idx
		low_q = idy
		low_r = idth
		low_dist = 9999
		for p in range(idx-1, idx+1):
			for q in range(idy-1, idy+1):
				for r in range(idth-1, idth+1):
					cur_dist = linalg.norm( np.array((x_k1, y_k1, th_k1%(2*math.pi))) - np.array((self.x_disc[p], self.y_disc[q], self.th_disc[r]%(2*math.pi))) )
					if (cur_dist < low_dist and self.mesh[p,q,r,0] < 40):
						low_dist = cur_dist
						low_p = p
						low_q = q
						low_r = r
		J_near = self.mesh[low_p,low_q,low_r,0]
		return J_near

	def interp_v(self, x_k1, y_k1, th_k1, idx, idy, idth):
		"""
		Interpolate the optimal control at a non gridpoint
		"""
		
		# v control
		# v = self.filter_edge_cases(idx, idy, idth, 1) # get valid neighbors for control v
		# v = [x for x in v if x != 9999] # get rid of unitialized values
		# c = Counter(v)
		# try:
		# 	most = c.most_common()[0][0] # select most common velocity
		# except:
		# 	print('MAKING A SELECTION')
		# 	most = self.v_disc[1]
		# v_star = most

		v_star = self.get_nearest_v(x_k1, y_k1, th_k1, idx, idy, idth)

		return v_star

	def get_nearest_v(self, x_k1, y_k1, th_k1, idx, idy, idth):
		"""
		Return the Euclidean nearest grid point that is initialized
		"""
		th_s = self.th_len
		if ((idx == 0) and (idy == 0)): # bottom left
			idx+=1
			idy+=1
		elif (idx >= self.x_len and (idy == 0)): # bottom right
			idx+=-1
			idy+=1
		elif (idy >= self.y_len and (idx == 0)): # top left
			idx+=1
			idy+=-1
		elif (idx >= self.x_len and idy >= self.y_len): # top right
			idx+=-1
			idy+=-1
		elif (idx == 0): # left
			idx+=1
		elif (idy == 0): # bottom
			idy+=1
		elif (idx >= self.x_len): # right
			idx+=-1
		elif (idy >= self.y_len): # top
			idy+=-1

		# # wrap around theta
		# if (idth >= self.th_len): 
		# 	low_id = self.th_disc[idth-1]
		# 	high_th = low_th + (self.th_disc[1] - self.th_disc[0])
		# elif (idth == 0):
		# 	low_th = self.th_disc[0] - (self.th_disc[1] - self.th_disc[0])
		# 	high_th = self.th_disc[idth] 
		# else:
		# 	low_th = self.th_disc[idth-1]
		# 	high_th = self.th_disc[idth] 

		low_p = idx
		low_q = idy
		low_r = idth%(self.th_len)
		low_dist = 9999
		for p in range(idx-1, idx+1):
			for q in range(idy-1, idy+1):
				for r in range(idth%(self.th_len)-1, idy%(self.th_len)+1):
					cur_dist = linalg.norm( np.array((x_k1, y_k1, th_k1%(2*math.pi))) - np.array((self.x_disc[p], self.y_disc[q], self.th_disc[r]%(2*math.pi))) )
					if (cur_dist < low_dist and self.mesh[p,q,r,1] != 9999):
						low_dist = cur_dist
						low_p = p
						low_q = q
						low_r = r

		J_near = self.mesh[low_p,low_q,low_r,1]
		print('near')
		print(J_near)
		return J_near

	def tri_interp(self, x_k1, y_k1, th_k1, idx, idy, idth, value):
		"""
		Interpolate trilinearly if all 8 corners available
		value: index of mesh stored valued to interpolate
		0=cost-to-go
		1=v
		2=phi
		"""
		# wrap around theta
		if (idth >= self.th_len): 
			low_th = self.th_disc[idth-1]
			high_th = low_th + (self.th_disc[1] - self.th_disc[0])
		elif (idth == 0):
			low_th = self.th_disc[0] - (self.th_disc[1] - self.th_disc[0])
			high_th = self.th_disc[idth] 
		else:
			low_th = self.th_disc[idth-1]
			high_th = self.th_disc[idth] 


		if (idth >= self.th_len): # wrap around theta
			values = np.array([self.mesh[idx-1:idx+1,idy-1:idy+1,idth-1,value], self.mesh[idx-1:idx+1,idy-1:idy+1,0,value]])
		elif (idth == 0):
			values = np.array([self.mesh[idx-1:idx+1,idy-1:idy+1,self.th_len-1,value], self.mesh[idx-1:idx+1,idy-1:idy+1,0,value]])
		else:	
			values=self.mesh[idx-1:idx+1,idy-1:idy+1,idth-1:idth+1,value]

		RGI = RegularGridInterpolator(points=[self.x_disc[idx-1:idx+1], self.y_disc[idy-1:idy+1], [low_th, high_th]],\
 							          values=values)
		output = RGI((x_k1, y_k1, th_k1))
		return output

	def tri_interp_control(self, x_k1, y_k1, th_k1, idx, idy, idth, value):
		"""
		Interpolate trilinearly if all 8 corners available
		value: index of mesh stored valued to interpolate
		0=cost-to-go
		1=v
		2=phi
		"""
		# wrap around theta
		if (idth >= self.th_len): 
			low_th = self.th_disc[idth-1]
			high_th = low_th + (self.th_disc[1] - self.th_disc[0])
		elif (idth == 0):
			low_th = self.th_disc[0] - (self.th_disc[1] - self.th_disc[0])
			high_th = self.th_disc[idth] 
		else:
			low_th = self.th_disc[idth-1]
			high_th = self.th_disc[idth] 

		if (idth >= self.th_len): # wrap around theta
			values = np.array([self.mesh[idx-1:idx+1,idy-1:idy+1,idth-1,value], self.mesh[idx-1:idx+1,idy-1:idy+1,0,value]])
		elif (idth == 0):
			values = np.array([self.mesh[idx-1:idx+1,idy-1:idy+1,self.th_len-1,value], self.mesh[idx-1:idx+1,idy-1:idy+1,0,value]])
		else:
			values=self.mesh[idx-1:idx+1,idy-1:idy+1,idth-1:idth+1,value]

		# get rid of any intruding initialized phi by averaging out the cell
		bad_indices = []
		good_vals = []
		new_values = copy.copy(values) # copy
		for index, val in np.ndenumerate(new_values):
			if val == 9999: # unitialized
				bad_indices.append(index)
			else:
				good_vals.append(val)
		for index in bad_indices:
			new_values[index] = np.mean(good_vals)

		RGI = RegularGridInterpolator(points=[self.x_disc[idx-1:idx+1], self.y_disc[idy-1:idy+1], [low_th, high_th]],\
 							 		  values=new_values)
		output = RGI((x_k1, y_k1, th_k1))
		return output


	def get_neighbors(self, x_k1, y_k1, th_k1):
		"""
		Get points in the 8-cube surrounding the X_k1 point
		"""
		idx = bisect.bisect(self.x_disc, x_k1) # this index and one below it are neighbors
		idy = bisect.bisect(self.y_disc, y_k1)
		idth = bisect.bisect(self.th_disc, th_k1)
		if (x_k1 == self.x_disc[self.x_len-1]): # knock down if right on the edge
			idx = idx-1
		if (y_k1 == self.y_disc[self.y_len-1]):
			idy = idy-1
		return idx, idy, idth

	def print_neighbors(self, x_k1, y_k1, th_k1):
		"""
		Print neighbor point indices
		"""
		print('x_k1, y_k1, th_k1')
		print(x_k1, y_k1, th_k1)
		idx, idy, idth = self.get_neighbors(x_k1, y_k1, th_k1)
		idth = idth%self.th_len # wrap theta around
		print('above neighbors')
		print(idx, idy, idth)
		print(self.x_disc[idx], self.y_disc[idy], self.th_disc[idth])
		print('below neighbors')
		print(idx-1, idy-1, idth-1)
		print(self.x_disc[idx-1], self.y_disc[idy-1], self.th_disc[idth-1])

	def filter_edge_cases(self, idx, idy, idth, var):
		"""
		Finds edge cases (out-of-bounds) and returns 8 nearest neighbors by adjusting into the mesh.
		"""
		th_s = self.th_len
		if ((idx == 0) and (idy == 0)): # bottom left
			print(idx, idy, idth)
			print('out')
			idx+=1
			idy+=1
		elif (idx >= self.x_len and (idy == 0)): # bottom right
			print(idx, idy, idth)
			print('out')
			idx+=-1
			idy+=1
		elif (idy >= self.y_len and (idx == 0)): # top left
			print(idx, idy, idth)
			print('out')
			idx+=1
			idy+=-1
		elif (idx >= self.x_len and idy >= self.y_len): # top right
			print(idx, idy, idth)
			print('out')
			idx+=-1
			idy+=-1
		elif (idx == 0): # left
			print(idx, idy, idth)
			print('out')
			idx+=1
		elif (idy == 0): # bottom
			print(idx, idy, idth)
			print('out')
			idy+=1
		elif (idx >= self.x_len): # right
			print(idx, idy, idth)
			print('out')
			idx+=-1
		elif (idy >= self.y_len): # top
			print(idx, idy, idth)
			print('out')
			idy+=-1
		
		# normal interp
		v000 = self.mesh[idx-1, idy-1, (idth-1)%th_s, var] # v_x_y_th
		v001 = self.mesh[idx-1, idy-1, (idth)%th_s, var]
		v010 = self.mesh[idx-1, idy, (idth-1)%th_s, var]
		v011 = self.mesh[idx-1, idy, (idth)%th_s, var]
		v100 = self.mesh[idx, idy-1, (idth-1)%th_s, var]
		v101 = self.mesh[idx, idy-1, (idth)%th_s, var]
		v110 = self.mesh[idx, idy, (idth-1)%th_s, var]
		v111 = self.mesh[idx, idy, (idth)%th_s, var]
		v = [v000, v001, v010, v011, v100, v101, v110, v111]
		return v

	def cost_to_move(self, x_k, y_k, th_k, x_k1, y_k1, th_k1):
		"""
		Calculate the additive cost of performing a control iteration
		"""
		g_k1 = self.v_disc[1] #np.linalg.norm(np.array((x_k, y_k)) - np.array((x_k1, y_k1)))
		return g_k1

	def find_u_opt(self, x_start, y_start, th_start):
		"""
		Returns the optimal control and state trajectory based on the computed dp mesh
		"""
		num = 0
		# Modified variables
		v_star = self.mesh[x_start, y_start, th_start, 1]   # optimal velocity
		phi_star = self.mesh[x_start, y_start, th_start, 2] # optimal steering angle
		x_k = self.x_disc[x_start]							   # x at step k
		y_k = self.y_disc[y_start]							   # y at step k
		th_k = self.th_disc[th_start]					   # theta at step k

		# Constants
		X_SPACING = self.x_disc[1]-self.x_disc[0]
		X_GOAL = self.x_disc[self.X0[0,0]]
		Y_GOAL = self.y_disc[self.X0[1,0]]
		TH_GOAL = self.th_disc[self.X0[2,0]]

		# State history for plotting
		x_k_hist = [x_k]
		y_k_hist = [y_k]
		th_k_hist = [th_k]

		print('x, y, th')
		print(x_k, y_k, th_k)
		print('control')
		print(v_star, phi_star)

		# While more than a grid space from the goal, keep searching
		while (np.linalg.norm( np.array((x_k, y_k, th_k%math.pi)) - np.array((X_GOAL, Y_GOAL, TH_GOAL)) ) > (X_SPACING)):
			num += 1
			if (num > 500):
				break
			if (v_star == np.inf or phi_star == np.inf):
				print('mesh error!')
				return 'blargh'
			#move
			x_k1, y_k1, th_k1 = self.f(x_k, y_k, th_k, v_star, phi_star)

			#interpolate to get u*
			print('x, y, th')
			print(x_k1, y_k1, th_k1)
			idx, idy, idth = self.get_neighbors(x_k1, y_k1, th_k1) #get valid neighbors for cost-to-go

			v_star = (self.v_disc[1]) * np.sign(self.tri_interp_control(x_k1, y_k1, th_k1, idx, idy, idth, 1))
			phi_star = self.tri_interp_control(x_k1, y_k1, th_k1, idx, idy, idth, 2)
			print('control')
			print(v_star, phi_star)

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

	def plot_3d_mesh(self, var):
		"""
		Plots the J for the mesh
		"""
		xid  = range(self.x_len)
		yid  = range(self.y_len)
		thid = range(self.th_len)
		# thid = [0]
		xpts = []
		ypts = []
		thpts = []
		Jpts = []
		for i in xid:
			for j in yid:
				for k in thid:
					xpts.append(self.x_disc[i])
					ypts.append(self.y_disc[j])
					thpts.append(self.th_disc[k])
					Jpts.append(self.mesh[i,j,k,var])
		ax2.scatter(xpts, ypts, thpts, zdir='z', c=Jpts, s=100)

	def plot_goal_state(self):
		"""
		Plots the car state of the goal
		"""
		# adjust car value
		plot_car = copy.deepcopy(self.goal_car)
		plot_car.x = self.x_disc[self.X0[0,0]]
		plot_car.y = self.y_disc[self.X0[1,0]]
		plot_car.th = self.th_disc[self.X0[2,0]]
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
                              interval=100, blit=True, init_func=self.init_anim_car)
		anim.save('anim6.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
		plt.show()
	#---Animate car parking of x, y, theta---------------------------------------------


def init_obstacles(scenario):
	"""
	Creates obstacles in the grid for collision checking; rectangular for now
	"""
	obstacles = []
	if (scenario == 0):
		obst1 = Obstacle(-.1,1.4,-.1, .9) #xmin, xmax, ymin, ymax
		obst2 = Obstacle(4.7,6.2,-.1, .9)
		obstacles.append(obst1)
		obstacles.append(obst2)

	if (scenario == 1):
		obst1 = Obstacle(0,1.2,0,1) #xmin, xmax, ymin, ymax
		obst2 = Obstacle(3.8,5,0,1)
		obstacles.append(obst1)
		obstacles.append(obst2)

	if (scenario == 2):
		obst1 = Obstacle(0,5,0,2.5) #xmin, xmax, ymin, ymax
		obst2 = Obstacle(10,15,0,2.5)
		obstacles.append(obst1)
		obstacles.append(obst2)

	if (scenario == 3):
		obst1 = Obstacle(1.9,3.1,1.9,3.1) #xmin, xmax, ymin, ymax
		obstacles.append(obst1)
	return obstacles

def init_car(scenario):
	"""
	Creates a car agent
	"""
	if (scenario == 0):
		# Car
		x0 = 5
		y0 = 5
		th0 = 0
		x1, y1 = -.2, .5
		x2, y2 =  x1, -y1
		x3, y3 = 1.2,  y1
		x4, y4 =  x3,  -y1

		car = Car(x0, y0, th0, x1, y1, x2, y2, x3, y3, x4, y4)
	return car

def init_mesh(scenario, car, obstacles):
	"""
	Creates a mesh instance
	"""
	if (scenario == 0):
		x_size = 12
		y_size = 12
		th_size = 12
		phi_size = 7 # must be odd for straight line driving
		phi_max = np.pi/4
		x_max = 5
		y_max = 5
		K_max = 11
		X0 = np.matrix([4,10,0]).T # goal state

	if (scenario == 1):
		x_size = 35
		y_size = 15
		th_size = 60
		phi_size = 19
		phi_max = np.pi/4
		x_max = 7
		y_max = 3
		K_max = 30
		X0 = np.matrix([14,2,0]).T # goal state

	if (scenario == 2): #~1.5 hours
		x_size = 30
		y_size = 30
		th_size = 30
		phi_size = 15
		phi_max = np.pi/4
		x_max = 5
		y_max = 5
		K_max = 45
		X0 = np.matrix([27,27,9]).T # goal state

	if (scenario == 3):
		x_size = 50
		y_size = 25
		th_size = 72
		phi_size = 30
		phi_max = np.pi/3
		x_max = 10
		y_max = 5
		K_max = 30
		X0 = np.matrix([2,10,2]).T # goal state

	mesh = Mesh(car, obstacles, x_size, y_size, th_size, phi_size, x_max, y_max, phi_max, X0, K_max)
	return mesh, x_max, y_max

if __name__ == '__main__':
	#Create a Reeds-Shepp instance
	START_STATE =        (3,3,23)
	obstacles =          init_obstacles(3)
	car =                init_car(0)
	start_car = 		 init_car(0)
	mesh, x_max, y_max = init_mesh(2, car, obstacles)	

	#---------------------------------------------
	# Init 2D Plot
	fig, ax = plt.subplots()
	ax.set_aspect(1)
	ax.set_xlim([0-1,x_max+1])
	ax.set_ylim([0-1,y_max+1])

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
	# start_car.set_car(mesh.x_disc[START_STATE[0]], mesh.y_disc[START_STATE[1]], mesh.th_disc[START_STATE[2]])
	# start_car.plot_car()
	for obstacle in obstacles:
		obstacle.plot_obstacle(fig)
	mesh.plot_mesh()
	mesh.plot_goal_state()
	# plt.show()
	# plt.close()		

	# Init 3D Plot
	fig2, ax2 = plt.subplots(subplot_kw=dict(projection='3d'))
	ax2.set_xlabel('x, m')
	ax2.set_ylabel('y, m')
	ax2.set_zlabel('theta, rad')

	# plt.close()
	#---------------------------------------------
	# Solve mesh using DP
	mesh.dp()
	mesh.plot_3d_mesh(0)
	print(mesh.mesh[0,5,0,:])
	# plt.show()

	# Plot u_opt history
	x_k_hist, y_k_hist, th_k_hist = mesh.find_u_opt(START_STATE[0], START_STATE[1], START_STATE[2])	
	mesh.show_anim_car(fig, x_k_hist, y_k_hist, th_k_hist)
	plt.show()
