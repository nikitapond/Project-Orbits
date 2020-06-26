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
ax1 = fig.add_subplot(151, projection='3d')
ax2 = fig.add_subplot(152, projection='3d')
ax3 = fig.add_subplot(153, projection='3d')
ax4 = fig.add_subplot(154, projection='3d')
ax5 = fig.add_subplot(155, projection='3d')

t = 10
time_span=np.linspace(0,t,t*10000)

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

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax1, show=False)
ax1.set_title("Stable butterfly system (" + str(t) + " years)")

solver.bodies.clear()
solver.AddBody(Body("1", 0.1, ip1, iv1*1.5))
solver.AddBody(Body("2", 0.1, ip2, iv2))
solver.AddBody(Body("3", 0.1, ip3, iv3))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax2, show=False, legend=False)
ax2.set_title(r"$v_{1_{new}}=1.5v_1$ - Stable (" + str(t) + " years)")

solver.bodies.clear()
solver.AddBody(Body("1", 0.1, ip1, iv1*3))
solver.AddBody(Body("2", 0.1, ip2, iv2))
solver.AddBody(Body("3", 0.1, ip3, iv3))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax3, show=False, legend=False)
ax3.set_title(r"$v_{1_{new}}=3v_1$ - Chaotic (" + str(t) + " years)")

t = 30
time_span=np.linspace(0,t,t*10000)

solver.bodies.clear()
solver.AddBody(Body("1", 0.15, ip1, iv1))
solver.AddBody(Body("2", 0.1, ip2, iv2))
solver.AddBody(Body("3", 0.1, ip3, iv3))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax4, show=False, legend=False)
ax4.set_title(r"$m_{1_{new}}=1.5m_1$ - Stable (" + str(t) + " years)")
solver.bodies.clear()

t = 10
time_span=np.linspace(0,t,t*10000)
solver.AddBody(Body("1", 0.18, ip1, iv1))
solver.AddBody(Body("2", 0.1, ip2, iv2))
solver.AddBody(Body("3", 0.1, ip3, iv3))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax5, show=False, legend=False)
ax5.set_title(r"$m_{1_{new}}=1.8m_1$ - Chaotic (" + str(t) + " years)")

fig.suptitle("Butterfly system with varying initial velocities and masses")
fig.savefig("butterfly_varying_mass_vel.png", dpi = fig.dpi)
plt.show()
