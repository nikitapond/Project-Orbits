import sys
sys.path.append('C:\\Users\\nikit\\AppData\\Local\\Programs\\Python\\python38\\lib\\site-packages')

import NBodyPlotter as nbp
from NBodyPlotter import NBodySolver
from NBodyPlotter import Body
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(figsize=(20,8))
time_span=np.linspace(0,1,10000)
ax1 = fig.add_subplot(141, projection='3d')
ax2 = fig.add_subplot(142, projection='3d')
ax3 = fig.add_subplot(143, projection='3d')
ax4 = fig.add_subplot(144, projection='3d')


au = 148070 #1 AU (relative to default scale value of 1000000m)
sun_mass = 1.989e6

solver = NBodySolver()

solver.AddBody(Body("sun",1.989e6, [0,0,0], [0,0,0]))
solver.AddBody(Body("earth",5.972,[148070, 0, 0], [0,29.72,0]))
solver.AddBody(Body("mercuary", 3.285e-1, [52868,0,0], [0,48,0]))
solver.AddBody(Body("venus", 4.867, [107630, 0, 0], [0,35,0]))
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax1, show=False)
ax1.set_title("Inner solar system - stable")

solver.bodies.clear()

mult = 2

solver.AddBody(Body("sun",1.989e6, [0,0,0], [0,0,0]))
solver.AddBody(Body("earth",5.972,[148070, 0, 0], [0,29.72,0]))
solver.AddBody(Body("mercuary", 3.285e-1, [52868,0,0], [0,48,0]))
solver.AddBody(Body("venus", 4.867, [107630, 0, 0], [0,35,0]))
solver.AddBody(Body("rogue star",1.989e6, [mult*au,-mult*au,0], [0,29.72,0]))

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax2, show=False, legend=False)
ax2.set_title("Inner solar system, \nrogue star at " + str(mult) + " au")

solver.bodies.clear()

mult = 3
time_span=np.linspace(0,4,40000)

solver.AddBody(Body("sun",1.989e6, [0,0,0], [0,0,0]))
solver.AddBody(Body("earth",5.972,[148070, 0, 0], [0,29.72,0]))
solver.AddBody(Body("mercuary", 3.285e-1, [52868,0,0], [0,48,0]))
solver.AddBody(Body("venus", 4.867, [107630, 0, 0], [0,35,0]))
solver.AddBody(Body("rogue star",1.989e6, [mult*au,-mult*au,0], [0,29.72,0]))

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax3, show=False, legend=False)
ax3.set_title("Inner solar system, \nrogue star at " + str(mult) + " au")

solver.bodies.clear()


mult = 5

solver.AddBody(Body("sun",1.989e6, [0,0,0], [0,0,0]))
solver.AddBody(Body("earth",5.972,[148070, 0, 0], [0,29.72,0]))
solver.AddBody(Body("mercuary", 3.285e-1, [52868,0,0], [0,48,0]))
solver.AddBody(Body("venus", 4.867, [107630, 0, 0], [0,35,0]))
solver.AddBody(Body("rogue star",1.989e6, [mult*au,-mult*au,0], [0,29.72,0]))

solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(ax=ax4, show=False, legend=False)
ax4.set_title("Inner solar system, \nrogue star at " + str(mult) + " au")

solver.bodies.clear()
fig.suptitle("Inner solar system with a rogue solar-mass \nstar appraoching at various distances")
fig.savefig("inner_solar_system_with_rogue.png", dpi=fig.dpi)
plt.show()
