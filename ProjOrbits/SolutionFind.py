import sys
sys.path.append('C:\\Users\\nikit\\AppData\\Local\\Programs\\Python\\python38\\lib\\site-packages')

import NBodyPlotter as nbp
from NBodyPlotter import NBodySolver
from NBodyPlotter import Body
import matplotlib.pyplot as plt
import numpy as np


solver = NBodySolver()

#Initially stable, reaches point of sudden instability (instabilty goes from
#   seeming stable, to completely chaotic within 1 orbital period)
# solver.AddBody(Body("sun",1e6, [-145000,0,0], [0,-10,0]))
# solver.AddBody(Body("second_sun",1e6,[145000, 0, 0], [0,10,0]))
# solver.AddBody(Body("third_sun", 1e6, [0,145000,0], [-10,0,0]))
# solver.AddBody(Body("third_sun", 1e6, [0,-145000,0], [10,0,0]))
# solver_name="4_suns"
# time_span=np.linspace(0,10,12000)



#Stable over long periods (order of magninutde differences in masses)

fig = plt.figure(figsize=(16,9))
solver_name="inner_solar_system_100yr_dif_earth_vel"
time_span=np.linspace(0,10,10000)
ax1 = fig.add_subplot(141, projection='3d')
ax2 = fig.add_subplot(142, projection='3d')
ax3 = fig.add_subplot(143, projection='3d')
ax4 = fig.add_subplot(144, projection='3d')

multiplier = 1

solver.AddBody(Body("sun",1.989e6, [0,0,0], [0,0,0]))
solver.AddBody(Body("earth",5.972,[148070, 0, 0], [0,29.72*multiplier,0]))
solver.AddBody(Body("mercuary", 3.285e-1, [52868,0,0], [0,48,0]))
solver.AddBody(Body("venus", 4.867, [107630, 0, 0], [0,35,0]))
solver.SolveNBodyProblem(time_span)
ax1.set_title("Inner solar system, \nearth velocity = " + str(multiplier) + " earth velocity")
solver.PlotNBodySolution(ax=ax1, show=False)

solver.bodies.clear()
multiplier = 1.2

solver.AddBody(Body("sun",1.989e6, [0,0,0], [0,0,0]))
solver.AddBody(Body("earth",5.972,[148070, 0, 0], [0,29.72*multiplier,0]))
solver.AddBody(Body("mercuary", 3.285e-1, [52868,0,0], [0,48,0]))
solver.AddBody(Body("venus", 4.867, [107630, 0, 0], [0,35,0]))
solver.SolveNBodyProblem(time_span)
ax2.set_title("Inner solar system, \nearth velocity = " + str(multiplier) + " earth velocity")
solver.PlotNBodySolution(ax=ax2, show=False)

solver.bodies.clear()
multiplier = 1.5

solver.AddBody(Body("sun",1.989e6, [0,0,0], [0,0,0]))
solver.AddBody(Body("earth",5.972,[148070, 0, 0], [0,29.72*multiplier,0]))
solver.AddBody(Body("mercuary", 3.285e-1, [52868,0,0], [0,48,0]))
solver.AddBody(Body("venus", 4.867, [107630, 0, 0], [0,35,0]))
solver.SolveNBodyProblem(time_span)
ax3.set_title("Inner solar system, \nearth velocity = " + str(multiplier) + " earth velocity")
solver.PlotNBodySolution(ax=ax3, show=False)

solver.bodies.clear()
multiplier = 2.0

solver.AddBody(Body("sun",1.989e6, [0,0,0], [0,0,0]))
solver.AddBody(Body("earth",5.972,[148070, 0, 0], [0,29.72*multiplier,0]))
solver.AddBody(Body("mercuary", 3.285e-1, [52868,0,0], [0,48,0]))
solver.AddBody(Body("venus", 4.867, [107630, 0, 0], [0,35,0]))
solver.SolveNBodyProblem(time_span)
ax4.set_title("Inner solar system, \nearth velocity = " + str(multiplier) + " earth velocity")
solver.PlotNBodySolution(ax=ax4, show=False)

fig.suptitle("Inner solar system with \nvarying earth velocities, \nevolved over 100 years\n")
fig.savefig(solver_name+".png", dpi=fig.dpi)

#
# #Figure 8, source http://homepages.math.uic.edu/~jan/mcs320s07/Project_Two/sol_body.html
# #Stability tested till 1000 solar years
# ip1 = [0.97000436* 100000, -0.24308753* 100000, 0]
# ip2 = [-ip1[0], -ip1[1], 0]
# ip3 = [0,0,0]
# iv3 = [-0.93240737*10,-0.86473146*10, 0]
# iv2 = [-iv3[0]/2, -iv3[1]/2, 0]
# iv1 = np.copy(iv2)
# solver.AddBody(Body("body1",1.5e5, ip1, iv1))
# solver.AddBody(Body("body2",1.5e5, ip2, iv2))
# solver.AddBody(Body("body3",1.5e5, ip3, iv3))
#
# solver_name="figure8_1000yr.png"


#solver.AddBody(Body("test1", 1e6, [0,0,0], [10,0,0]))
#solver.AddBody(Body("test2", 1e6, [0,1000000,0], [15,0,0]))
# solver_name="ug"

# solver.AddBody(Body("star_1", 1e6, [0,-100000,0], [ 15, 0, 0]))
# solver.AddBody(Body("star_2", 1e6, [0, 100000,0], [-15, 0, 0]))
# solver.AddBody(Body("star_2", 1e6, [0, 0,0], [0, 0, 5]))
#
# time_span=np.linspace(0,3,12000)
#
# #time_span=np.linspace(0,5,12000)
# solver.SolveNBodyProblem(time_span)
# solver.AnimateNBodySolution()
# solver.PlotNBodySolution(saveFile=solver_name)
# solver = NBodySolver()
#
#
# solver.AddBody(Body("star_1", 1e6, [0,-100000,0], [ 15, 0, 0]))
# solver.AddBody(Body("star_2", 1e6, [0, 100000,0], [-15, 0, 0]))
# solver.AddBody(Body("star_2", 1e6, [0, 0,1000], [0, 0, 0]))
#
# fig = plt.figure(figsize=(15,5))
# ax1 = fig.add_subplot(141, projection='3d')
# ax2 = fig.add_subplot(142, projection='3d')
# ax3 = fig.add_subplot(143, projection='3d')
# ax4 = fig.add_subplot(144, projection='3d')
# #
# time_span=np.linspace(0,3,3000) #1000 time spacing per solar year
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax1, show=False)
#
# time_span=np.linspace(0,3,6000) #2000 time spacing per solar year
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax2, show=False)
#
# time_span=np.linspace(0,3,12000) #2000 time spacing per solar year
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax3, show=False)
#
# time_span=np.linspace(0,3,18000) #2000 time spacing per solar year
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax4, show=False)
#
# definition = 250000 #Number of points per solar year
#
# time_span=np.linspace(0,3,1*definition) #1000 time spacing per solar year
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax1, show=False)
# ax1.set_title("1 years")
#
# time_span=np.linspace(0,3,3*definition) #2000 time spacing per solar year
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax2, show=False)
# ax2.set_title("3 years")
#
# time_span=np.linspace(0,5, 5*definition) #2000 time spacing per solar year
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax3, show=False)
# ax3.set_title("5 years")
#
# time_span=np.linspace(0,10,10*definition) #2000 time spacing per solar year
# solver.SolveNBodyProblem(time_span)
# solver.PlotNBodySolution(ax=ax4, show=False)
# ax4.set_title("10 years")
#
# fig.suptitle(str(definition) + " data points per solar year")
#
# fig.savefig("interesting3_longer_time_"+str(definition)+".png", dpi=fig.dpi)

plt.show()
