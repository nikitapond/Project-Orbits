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

star_vel = np.sqrt(nbp.G *mass_scale/(dist_scale))/(vel_scal*2)

def Reset(solver):
    solver.bodies.clear()
    solver.AddBody(Body("star 1", 1, [0, -1, 0], [star_vel,0,0]))
    solver.AddBody(Body("star 2", 1, [0, 1, 0], [-star_vel,0,0]))
    solver.AddBody(Body("mid boi", 0.1, [0, 0, 1], [0,0,0]))
Reset(solver)

t = 10
time_span=np.linspace(0,t,t*10000)

#rBodyDist = 1e30
#solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
solver.SolveNBodyProblem(time_span)
#fig = plt.figure()
#ax =fig.add_subplot(111, projection='3d')

#solver.PlotNBodySolution(ax, legend=False)


fig = plt.figure(figsize=(13, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
#ax4 = fig.add_subplot(144, projection='3d')

#star_vel = np.sqrt(nbp.G *mass_scale/(5*dist_scale))/(vel_scal*2)

t = 10
time_span=np.linspace(0,t,t*10000)
#Initiate solver

solver.SetSolverRelativeValues(mass_scale, dist_scale, vel_scal, orbit_period)

# print(star_vel)
# def Reset(solver):
#     solver.bodies.clear()
#     solver.AddBody(Body("star 1", 1, [0, -5, 0], [star_vel,0,0]))
#     solver.AddBody(Body("star 2", 1, [0, 5, 0], [-star_vel,0,0]))
#     solver.AddBody(Body("mid boi", 0.1, [0, 0, 1], [0,0,0]))

# Reset(solver)
# solver.SolveNBodyProblem(time_span)
# #solver.AnimateNBodySolution()
# solver.PlotNBodySolution(ax=ax1, show=False, legend=False)
# ax1.set_title("Halo system")

#
# Reset(solver)
#
#
#
Reset(solver)

rBodyDist = 1e45
solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax1, show=False, legend=False)
ax1.set_title("Halo system\n"+r"rogue body at $R_{rs}=$" + str(rBodyDist))

Reset(solver)
rBodyDist = 1e44
solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax2, show=False, legend=True)
ax2.set_title("Halo system\n rogue body at\n"+r"$R_{rs}=$" + str(rBodyDist))

fig.subplots_adjust(wspace=0.15)
# Reset(solver)
#
# t = 500
# time_span=np.linspace(0,t,t*10000)
#
# rBodyDist = 4.5e45
# solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax4, show=False, legend=False)
# ax4.set_title("halo system\n"+r"rogue body at $R_{rs}=$" + str(rBodyDist))

plt.show()
