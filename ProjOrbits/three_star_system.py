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
vel_scal = 1000 #m/s (such that inputted units are in Km/s)
orbit_period = 356*24*60*60 #s

t = 20
time_span=np.linspace(0,t,t*1000)
#Create figures
fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(151, projection='3d')
ax2 = fig.add_subplot(152, projection='3d')
ax3 = fig.add_subplot(153, projection='3d')
ax4 = fig.add_subplot(154, projection='3d')
ax5 = fig.add_subplot(155, projection='3d')

#Initiate solver
solver = NBodySolver()
solver.SetSolverRelativeValues(mass_scale, dist_scale, vel_scal, orbit_period)

#Define initial values for 3 bodies
ip1 = [0.97000436, -0.24308753, 0]
ip2 = [-ip1[0], -ip1[1], 0]
ip3 = [0,0,0]
iv3 = [-0.93240737*10,-0.86473146*10, 0]
iv2 = [-iv3[0]/2, -iv3[1]/2, 0]
iv1 = np.copy(iv2)

#Create and plot stable figure of 8 orbit
solver.AddBody(Body("body1",1, ip1, iv1))
solver.AddBody(Body("body2",1, ip2, iv2))
solver.AddBody(Body("body3",1, ip3, iv3))

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax1, show=False)
ax1.set_title("Stable figure of 8 orbit\n")

solver.bodies.clear()
#Create and plot figure of 8 with slight change to v3
solver.AddBody(Body("body1",1, ip1, iv1))
solver.AddBody(Body("body2",1, ip2, iv2))
solver.AddBody(Body("body3",1, ip3, np.array(iv3)*1.1))

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax2, show=False, legend=False)
ax2.set_title(r"$v_{3_{new}} =1.1v_{3}$ - Stable")

solver.bodies.clear()
#Create and plot figure of 8 with larger change to v3
solver.AddBody(Body("body1",1, ip1, iv1))
solver.AddBody(Body("body2",1, ip2, iv2))
solver.AddBody(Body("body3",1, ip3, np.array(iv3)*1.2))

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax3, show=False, legend=False)
ax3.set_title(r"$v_{3_{new}} =1.2v_{3}$ - Chaotic")

solver.bodies.clear()
#Create and plot figure of 8 with slight change to m3
solver.AddBody(Body("body1",1, ip1, iv1))
solver.AddBody(Body("body2",1, ip2, iv2))
solver.AddBody(Body("body3",1.1, ip3, iv3))

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax4, show=False, legend=False)
ax4.set_title(r"$m_{3_{new}} =1.1m_{3}$ - Stable")

solver.bodies.clear()
#Create and plot figure of 8 with larger change to m3
solver.AddBody(Body("body1",1, ip1, iv1))
solver.AddBody(Body("body2",1, ip2, iv2))
solver.AddBody(Body("body3",1.2, ip3, iv3))

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax5, show=False, legend=False)
ax5.set_title(r"$m_{3_{new}} =1.2m_{3}$ - Chaotic")
fig.suptitle("Varying initial masses and velocities of the figure of 8 stable system")
fig.savefig("figure_8_varying_mass_and_vel.png", dpi=fig.dpi)

plt.show()
