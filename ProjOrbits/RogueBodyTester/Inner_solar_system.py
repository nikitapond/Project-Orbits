import sys
sys.path.append('C:\\Users\\nikit\\AppData\\Local\\Programs\\Python\\python38\\lib\\site-packages')

import NBodyPlotter as nbp
from NBodyPlotter import NBodySolver
from NBodyPlotter import Body
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(25,8))
time_span=np.linspace(0,2,10000)
ax1 = fig.add_subplot(141, projection='3d')
ax2 = fig.add_subplot(142, projection='3d')
ax3 = fig.add_subplot(143, projection='3d')
ax4 = fig.add_subplot(144, projection='3d')
multScale = 1e4

au = 148070 #1 AU (relative to default scale value of 1000000m)
sun_mass = 1.989e6
solver = NBodySolver()

#
# Mass_scale = 1e24 #Kg
# Distance_scale = 1000000 #m (1e6)
# Velocity_scale = 1000 #m/s (1e3)
# Time_scale = 365.25*24*3600 #s #orbital period of earth

solver.SetSolverRelativeValues(nbp.Mass_scale, nbp.Distance_scale*multScale, nbp.Velocity_scale, nbp.Time_scale)
#Create stable system
solver.AddBody(Body("Sun",1.989e6, [0,0,0], [0,0,0]))
solver.AddBody(Body("Earth",5.972,[148070/multScale, 0, 0], [0,29.72,0]))
solver.AddBody(Body("Mercury", 3.285e-1, [52868/multScale,0,0], [0,48,0]))
solver.AddBody(Body("Venus", 4.867, [107630/multScale, 0, 0], [0,35,0]))
#plot stable system
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax1, show=False, legend=False)
ax1.set_title("Inner solar system\nstable")

rBodyDist = 3

#Add rogue body
solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, nbp.Mass_scale, nbp.Distance_scale*multScale, nbp.Velocity_scale))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax2, show=False, legend=False)
ax2.set_title("Inner solar system\n"+r"rogue body at $R_{rs}=$" + str(rBodyDist))

#reset stable system
solver.bodies.clear()
solver.AddBody(Body("Sun",1.989e6, [0,0,0], [0,0,0]))
solver.AddBody(Body("Earth",5.972,[148070/multScale, 0, 0], [0,29.72,0]))
solver.AddBody(Body("Mercury", 3.285e-1, [52868/multScale,0,0], [0,48,0]))
solver.AddBody(Body("Venus", 4.867, [107630/multScale, 0, 0], [0,35,0]))
rBodyDist = 2.5

#Add rogue body
solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, nbp.Mass_scale, nbp.Distance_scale*multScale, nbp.Velocity_scale))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax3, show=False, legend=False)
ax3.set_title("Inner solar system\n"+r"rogue body at $R_{rs}=$" + str(rBodyDist))


#reset stable system
solver.bodies.clear()
solver.AddBody(Body("Sun",1.989e6, [0,0,0], [0,0,0]))
solver.AddBody(Body("Earth",5.972,[148070/multScale, 0, 0], [0,29.72,0]))
solver.AddBody(Body("Mercury", 3.285e-1, [52868/multScale,0,0], [0,48,0]))
solver.AddBody(Body("Venus", 4.867, [107630/multScale, 0, 0], [0,35,0]))
rBodyDist = 2.1

#Add rogue body
solver.AddBody(nbp.CreateRogueBody(solver.bodies, rBodyDist, nbp.Mass_scale, nbp.Distance_scale*multScale, nbp.Velocity_scale))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax4, show=False)
ax4.set_title("Inner solar     \n system rogue    \n body at     \n"+r"$R_{rs}=$" + str(rBodyDist) + "     ")

#solver.PlotNBodySolution(ax=ax2, show=False)
#ax2.set_title("Inner solar system - stable")
fig.subplots_adjust(wspace=0.1)
plt.show()
