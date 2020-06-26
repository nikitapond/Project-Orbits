import sys
sys.path.append('C:\\Users\\nikit\\AppData\\Local\\Programs\\Python\\python38\\lib\\site-packages')

import NBodyPlotter as nbp
from NBodyPlotter import NBodySolver
from NBodyPlotter import Body
import matplotlib.pyplot as plt
import numpy as np

#Define scale values to keep close to unity
mass_scale = 1.5e29 #Kg
dist_scale = 1e11 #m
vel_scal = 10000 #m/s (such that inputted units are in Km/s)
orbit_period = 356*24*60*60 #s

t = 10
time_span=np.linspace(0,t,t*1000)
#
# #Create figures
# fig = plt.figure(figsize=(20,8))
# ax1 = fig.add_subplot(141, projection='3d')
# ax2 = fig.add_subplot(142, projection='3d')
# ax3 = fig.add_subplot(143, projection='3d')
# ax4 = fig.add_subplot(144, projection='3d')
# #
#Initiate solver
solver = NBodySolver()
solver.SetSolverRelativeValues(mass_scale, dist_scale, vel_scal, orbit_period)

#Define initial values for 3 bodies
ip1 = [0.97000436, -0.24308753, 0]
ip2 = [-ip1[0], -ip1[1], 0]
ip3 = [0,0,0]
iv3 = [-0.93240737,-0.86473146, 0]
iv2 = [-iv3[0]/2, -iv3[1]/2, 0]
iv1 = np.copy(iv2)


def Reset(solver):
    solver.bodies.clear()
    #Create and plot stable figure of 8 orbit
    solver.AddBody(Body("Body 1",1, ip1, iv1))
    solver.AddBody(Body("Body 2",1, ip2, iv2))
    solver.AddBody(Body("Body 3",1, ip3, iv3))
#
# # #Set stable figure 8 and plot
# Reset(solver)
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax1, show=False, legend=False)
# ax1.set_title("Stable figure of 8 orbit\n")
#
# Reset(solver)
# rBodyDist = 5
#
# solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax2, show=False, legend=False)
# ax2.set_title("Figure 8 system\n"+r"rogue body at $R_{rs}=$" + str(rBodyDist))
#
# Reset(solver)
# rBodyDist = 3
#
# solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax3, show=False, legend=False)
# ax3.set_title("Figure 8 system\n"+r"rogue body at $R_{rs}=$" + str(rBodyDist))
#
# Reset(solver)
# rBodyDist = 2.6
#
# solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax4, show=False)
# ax4.set_title("Figure 8      \n system rogue      \nbody at        \n" + r"$R_{rs}=$" + str(rBodyDist) + "         ")
# fig.subplots_adjust(wspace=0.15)
#
# #Create figures
fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

Reset(solver)
rBodyDist = 7
solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
solver.SolveNBodyProblem(time_span)

solver.PlotNBodySolution(ax=ax1, show=False, legend=False)
ax1.set_title("Figure 8 system\n"+r"rogue body at $R_{rs}=$" + str(rBodyDist))

t = 75
time_span=np.linspace(0,t,t*1000)

Reset(solver)
rBodyDist = 5

solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax2, show=False, legend=False)
ax2.set_title("Figure 8 system\n"+r"rogue body at $R_{rs}=$" + str(rBodyDist))

Reset(solver)
rBodyDist = 3

solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax3, show=False, legend=True)
ax3.set_title("Figure 8 system\n rogue body at\n"+r"$R_{rs}=$" + str(rBodyDist))

plt.show()
