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


#Create figures
fig = plt.figure(figsize=(20,8))
fig.subplots_adjust(wspace=0.0)
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')
#ax4 = fig.add_subplot(144, projection='3d')
t = 10
time_span=np.linspace(0,t,t*1000)

ip1 = np.array([-1,0,0])
ip2 = -ip1
ip3 = np.array([0,0,0])

vx1 = 0.30689 * 10
vy1 = 0.12551 * 10
iv1 = np.array([vx1,vy1, 0])
iv2 = np.array([vx1, vy1, 0])
iv3 = np.array([-2*vx1, -2*vy1, 0])

def Reset(solver):
    solver.bodies.clear()
    solver.AddBody(Body("Body 1", 0.1, ip1, iv1))
    solver.AddBody(Body("Body 2", 0.1, ip2, iv2))
    solver.AddBody(Body("Body 3", 0.1, ip3, iv3))
#
Reset(solver)
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax1, show=False, legend=False)
ax1.set_title("Stable Butterfly orbit\n")


Reset(solver)
rBodyDist = 7
solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax2, show=False, legend=False)
ax2.set_title("Butterfly system\n"+r"rogue body at $R_{rs}=$" + str(rBodyDist)+ " (10 yr)")

# t = 15
# time_span=np.linspace(0,t,t*1000)
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax3, show=False, legend=False)
# ax3.set_title("butterfly system\n"+r"rogue body at $R_{rs}=$" + str(rBodyDist)+ " (15 yr)")
t = 16
time_span=np.linspace(0,t,t*1000)
Reset(solver)
rBodyDist = 1e100

b = nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal)
print(b.startPos)

solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax3, show=False, legend=True)
ax3.set_title("Butterfly system\n rogue body at \n"+r"$R_{rs}=$" + str(rBodyDist) + " (16 yr)")
#
# Reset(solver)
# rBodyDist = 3
# solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax3, show=False, legend=False)
# ax3.set_title("butterfly system\n"+r"rogue body at $R_{rs}=$" + str(rBodyDist))
#
# Reset(solver)
# rBodyDist = 2.5
# solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, mass_scale, dist_scale, vel_scal))
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax4, show=False, legend=True)
# ax4.set_title("butterfly system\n"+r"rogue body at $R_{rs}=$" + str(rBodyDist))
plt.show()
