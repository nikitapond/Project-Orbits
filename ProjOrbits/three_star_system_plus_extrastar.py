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

t = 30
time_span=np.linspace(0,t,t*1000)
#Create figures
fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(141, projection='3d')
ax2 = fig.add_subplot(142, projection='3d')
ax3 = fig.add_subplot(143, projection='3d')
ax4 = fig.add_subplot(144, projection='3d')

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

#Create stable figure of 8 orbit
solver.AddBody(Body("body1",1, ip1, iv1))
solver.AddBody(Body("body2",1, ip2, iv2))
solver.AddBody(Body("body3",1, ip3, iv3))

#Add rogue star to system, simulate over 30 and 60 years
dist_away = 5
solver.AddBody(Body("rogue body",1, [dist_away,0,0], [0,10,0]))

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax1, show=False)
ax1.set_title("Figure of 8 orbit with \nrogue body at distance of \n" + str(dist_away) + " over time span of " +str(t) + " years")


t = 60
time_span=np.linspace(0,t,t*1000)
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax2, show=False)
ax2.set_title("Figure of 8 orbit with \nrogue body at distance of \n" + str(dist_away) + " over time span of " +str(t) + " years")

solver.bodies.clear()
#Create stable figure of 8 orbit
solver.AddBody(Body("body1",1, ip1, iv1))
solver.AddBody(Body("body2",1, ip2, iv2))
solver.AddBody(Body("body3",1, ip3, iv3))

t = 200
time_span=np.linspace(0,t,t*1000)
#Add rogue star to system, simulate over 30 and 60 years
dist_away = 6
solver.AddBody(Body("rogue body",1, [dist_away,0,0], [0,10,0]))

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax3, show=False)
ax3.set_title("Figure of 8 orbit with \nrogue body at distance of \n" + str(dist_away) + " over time span of " +str(t) + " years")
#
solver.bodies.clear()
#Create stable figure of 8 orbit
solver.AddBody(Body("body1",1, ip1, iv1))
solver.AddBody(Body("body2",1, ip2, iv2))
solver.AddBody(Body("body3",1, ip3, iv3))
dist_away = 6
solver.AddBody(Body("rogue body",1, [dist_away,0,0], [0,25,0]))

t = 10
time_span=np.linspace(0,t,t*1000)
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax4, show=False)
ax4.set_title("Figure of 8 orbit with \nrogue body at distance of \n" + str(dist_away) + " over time span of " +str(t) + " years\n with larger rogue body velocity")
plt.show()

fig.savefig("figure_8_rogue_body.png", dpi=fig.dpi)
#solver.bodies.clear()
