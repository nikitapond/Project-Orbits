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
fig = plt.figure(figsize=(14,7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')


t = 25
time_span=np.linspace(0,t,t*100000)

ip1 = np.array([-1,0,0])
ip2 = -ip1
ip3 = np.array([0,0,0])

vx1 = 0.30689 * 10
vy1 = 0.12551 * 10
iv1 = np.array([vx1,vy1, 0])
iv2 = np.array([vx1, vy1, 0])
iv3 = np.array([-2*vx1, -2*vy1, 0])

solver.AddBody(Body("1", 0.1, ip1, iv1))
solver.AddBody(Body("2", 0.1, ip2, iv2))
solver.AddBody(Body("3", 0.1, ip3, iv3))

dist = 14
solver.AddBody(Body("rogue body", 0.1, [dist,0, 0], [0, 7, 0]))


solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax1, show=False)
ax1.set_title("Stable butterfly system with rogue \nbody at relative distance of " + str(dist))

solver.bodies.clear()
solver.AddBody(Body("1", 0.1, ip1, iv1))
solver.AddBody(Body("2", 0.1, ip2, iv2))
solver.AddBody(Body("3", 0.1, ip3, iv3))

dist = 10
solver.AddBody(Body("rogue body", 0.1, [dist,0, 0], [0, 7, 0]))

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax2, show=False)
ax2.set_title("Stable butterfly system with rogue \nbody at relative distance of " + str(dist))

fig.suptitle("Butteryfly system with rogue body at varying distances")
fig.savefig("butteryfly_rogue.png", dpi=fig.dpi)

plt.show()
