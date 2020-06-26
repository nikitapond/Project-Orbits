import sys
sys.path.append('C:\\Users\\nikit\\AppData\\Local\\Programs\\Python\\python38\\lib\\site-packages')

import NBodyPlotter as nbp
from NBodyPlotter import NBodySolver
from NBodyPlotter import Body
import matplotlib.pyplot as plt
import numpy as np

#Define scale values to keep close to unity
mass_scale = 1e30 #Kg
dist_scale = 1e11 #m
vel_scal = 1000 #m/s (such that inputted units are in Km/s)
orbit_period = 356*24*60*60 #s

solver = NBodySolver()
solver.SetSolverRelativeValues(mass_scale, dist_scale, vel_scal, orbit_period)
fig = plt.figure(figsize=(15, 15))

ax = []

for i in range(12):
    ax.append(fig.add_subplot(3,4,i+1, projection='3d'))

#
# star_vel = np.sqrt(nbp.G * 1*mass_scale/(dist_scale))/(vel_scal*2)
#
# t = 30
# time_span=np.linspace(0,t,t*1000)
# #Initiate solver
#
# solver.SetSolverRelativeValues(mass_scale, dist_scale, vel_scal, orbit_period)
# solver.AddBody(Body("star 1", 1, [-1, 0, 0], [0,star_vel,0]))
# solver.AddBody(Body("star 2", 1, [1, 0, 0], [0,-star_vel,0]))
# solver.SolveNBodyProblem(time_span)
# solver.AnimateNBodySolution()

#
star_vel = np.sqrt(nbp.G *mass_scale/(dist_scale))/(vel_scal*2)

t = 10
time_span=np.linspace(0,t,t*100000)
#Initiate solver

solver.SetSolverRelativeValues(mass_scale, dist_scale, vel_scal, orbit_period)
solver.AddBody(Body("star 1", 1, [0, -1, 0], [star_vel,0,0]))
solver.AddBody(Body("star 2", 1, [0, 1, 0], [-star_vel,0,0]))
solver.AddBody(Body("mid boi", 0.1, [0, 0, 1], [0,0,0]))
#
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax[0])
ax[0].set_title("Halo system 1 after " + str(t) + " solar years")

#solver.AnimateNBodySolution()
