import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from scipy import integrate

G=6.67408e-11 #m^3Kg^-1s^-1 #Big G
"""
Below are the default values for scales of mass, distance, velocity, as well
as the time scale
"""
Mass_scale = 1e24 #Kg
Distance_scale = 1000000 #m (1e6)
Velocity_scale = 1000 #m/s (1e3)
Time_scale = 365.25*24*3600 #s #orbital period of earth

colours = ['r','g','b','y','pink', 'c', 'm'] #Colours used for plotting

def SetRelativeValues(mass_scale, dist_scale, vel_scale, time_scale):
    """
    Calculates constants used by the differential equation solver based off
    the given values.
    Inputs:
        mass_scale (M)- The scale unit for mass (kg) that all object masses
                    are relative to
        dist_scale (D)- The scale unit for distance (m) that all distances are
                    relative to
        vel_scale (V)- The scale unit for velocity (m/s) that all velocities are
                    relative to
        time_scale (T)- The time span (s) that a value of 1 in the time span
                    used in scipy.integrare.odeint represents
    Outputs:
        k1 - The product (G*T*M)/(D^2 * V), constant to find dr/dt in ODE solver
        k2 - The product (V*T)/D, constant to find dv/dt
    """
    k1 = G*time_scale*mass_scale/(dist_scale**2 * vel_scale)
    k2 = vel_scale*time_scale/dist_scale
    return k1,k2

def SetDefaultReatives():
    """
    Calculates constants used by the differential equation solver based off
    the default values.
    Default values:
        Mass_scale = 10e23 Kg, Distance_scale = 10e5 m, Velocity_scale = 1000 m/s,
        Time_scale = 365.25*24*3600 s, 1 orbit for Earth
    Outputs:
        k1 - The product (G*T*M)/(D^2 * V), constant to find dr/dt in ODE solver
        k2 - The product (V*T)/D, constant to find dv/dt
    """
    k1 = G*Time_scale*Mass_scale/(Distance_scale**2 * Velocity_scale)
    k2 = Velocity_scale*Time_scale/Distance_scale
    return k1,k2
class NBodySolver:
    def __init__(self):
        """
        Initialises the NBodySolver
        """
        #Set up list of bodies
        self.bodies = []
        #Set constants as default, set solved to False
        self.k1, self.k2 = SetDefaultReatives()
        self.solved=False
        self.dist_scale = Distance_scale


    def AddBody(self, body):
        """
        Adds the supplied body to this solver.
        Inputs:
            Body - the body to add to this solver
        """
        self.bodies.append(body)

    def AddNewBody(self, name, mass, position, velocity):
        """
        Creates a new Body based on the given arguments, and then adds it to
        this solver.
        Inputs:
            name - The name of the body
            mass - The mass of the body relative to the Mass_scale
            position - The initial position of the body relative to Distance_scale
            velocity - The initial velocity of the body relative to Velocity_scale
        """
        self.bodies.append(Body(name, mass, position, velocity))

    def SetSolverRelativeValues(self, mass_scale, dist_scale, vel_scale, time_scale):
        """
        Calculates constants used by the differential equation solver based off
        the given values.
        Inputs:
            mass_scale (M)- The scale unit for mass (kg) that all object masses
                        are relative to
            dist_scale (D)- The scale unit for distance (m) that all distances are
                        relative to
            vel_scale (V)- The scale unit for velocity (m/s) that all velocities are
                        relative to
            time_scale (T)- The time span (s) that a value of 1 in the time span
                        used in scipy.integrare.odeint represents
        """
        self.k1,self.k2 = SetRelativeValues(mass_scale, dist_scale, vel_scale, time_scale)
        self.dist_scale = dist_scale


    def SolveNBodyProblem(self, time_span):
        """
        Prepairs the bodies of this solver, ready to be added to the ODE solver.
        Extracts the relevent data from the result, and saves to the object.
        Inputs:
            time_span - The time span that the simulation should be run over, a
                time span of 1 represents 1 Time_scale
            shouldCM - bool value, if true all values will be from the centre of
                mass reference frame
        """
        self.time_span = time_span
        initial, masses = PrepairValues(self.bodies)
        self.N = len(self.bodies)
        n_body_sol = integrate.odeint(CoupledNBodyODE, initial, time_span, args=(self.k1,self.k2,self.N, [masses]))
        #Create array of just the positions of the solution
        self.bodySol = []
        for i in range(self.N):
            self.bodySol.append(n_body_sol[:,(3*i):(3*(i+1))])
        self.solved=True
        #Return both the neat solution, as well as the full solution
    def PlotNBodySolution(self, ax=None, show=True, saveFile=None, legend=True):
        if(self.solved==False):
            print("Solution must be found before plotting. Use NBodySolver.SolveNBodyProblem")
            return

        if(ax==None):
            fig=plt.figure(figsize=(5,5))
            ax=fig.add_subplot(111,projection="3d")
        #Iterate each body and extract full path for each body, then plot.
        for i in range(self.N):
            b = self.bodySol[i]
            ax.scatter(b[0,0], b[0,1], b[0,2], marker='o', color=colours[i])
            ax.plot(b[:,0],b[:,1],b[:,2],color=colours[i], label=self.bodies[i].name)
            ax.scatter(b[-1,0], b[-1,1], b[-1,2], marker='*', color=colours[i])
        #Add details to plot, then show
        dim = r"$\times 10^{"+str(np.log10(self.dist_scale))+"}$m"
        #Add details to plot, then show
        ax.set_xlabel("X " + dim,fontsize=12)
        ax.set_ylabel("Y " + dim,fontsize=12)
        ax.set_zlabel("Z " + dim,fontsize=12)


        if(legend):
            ax.legend(loc="upper right",fontsize=10)
        if(show):
            plt.show()
        if(saveFile != None):
            fig.savefig(saveFile, dpi=fig.dpi)
    def AnimateNBodySolution(self, axis_size=None):
        if(self.solved==False):
            print("Solution must be found before animating. Use NBodySolver.SolveNBodyProblem")
            return
        data = []
        for i in range(self.N):
            data.append([self.bodySol[i][:,0],self.bodySol[i][:,1],self.bodySol[i][:,2]])
        #Turn to numpy array
        data = np.array(data)
        #Check if axis_size is defined. If not, define it
        if(axis_size==None):
            axis_size = np.max(self.bodySol)
        #Create 3d figure to plot on
        fig=plt.figure(figsize=(6,6))
        ax=fig.add_subplot(111,projection="3d")
        #Extract data into a set of 3 dimensional lines
        lines = [ax.plot(dat[0,0:1],dat[1,0:1],dat[2,0:1], label=self.bodies[i].name)[0] for i, dat in enumerate(data)]
        for i, line in enumerate(lines):
            line.set_color(colours[i])

            #line.label("test" + str(i))

        def update_lines(num, dataLines, lines):
            """
            Update function for the animation.
            Inputs:
                num - the current iteration of the animation
                dataLines - all of the 3d position solutions of all bodies
                lines - the lines to animate
            """

            #i=0
            for line, data in zip(lines, dataLines):
                #line.set_color(colours[i])
                line.set_data(data[0:2, :num])
                line.set_3d_properties(data[2, :num])

                #i+=1
            return lines
        ax.legend(loc="upper left",fontsize=14)
        #Set up axis of plot
        ax.set_xlim3d([-axis_size,axis_size])
        ax.set_xlabel('X')

        ax.set_ylim3d([-axis_size,axis_size])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-axis_size,axis_size])
        ax.set_zlabel('Z')
        #Create animation, then show to user
        line_ani = FuncAnimation(fig, update_lines, len(self.time_span), fargs=(data, lines),
                                       interval=0.1, blit=True, repeat=False)
        plt.show()

class Body:
    """
    class that holds details for each body
    """
    def __init__(self, name, mass, startPos, startVel):
        """
        Initiates the Body with the supplied values
        Inputs:
            name - The name of the body
            mass - The mass of the body relative to the Mass_scale supplied to
                the N body solver
            startPos - Array like (3 dimensions, [x,y,z]). Position relative to
                the Distance_scale supplied to the N body solver
            startVel - Array like (3 dimensions, [v_x, v_y, v_z]). Velocity
                relative to the Velocity_Scale supplied to the N body solver.
        """
        self.name=name
        self.mass=mass
        self.startPos=startPos
        self.startVel=startVel


def CoupledNBodyODE(rv, t, k1, k2, N, masses):
    """
    Calculates the new velocity and position of each of the bodies at the given
    iteration.
    Inputs:
        rv - array like, holds position and velocity of all bodies
        t - supplied for function to work with scipy.integrate.odeint, unused
        k1 - constant calculated based on scale values, used to find velocity
            of each body
        k2 - constant calculated based on scale values, used to find acceleration
            of each body
        masses - array like, mass of each of the bodies
        shouldCM - bool value, if true all values will be from the centre of
            mass reference frame
    Outputs - flat 1d array of floats, represents position and velocity of all
            bodies. Same format as input 'rv'
    """
    #Prepair arrays to hold positions, velocities, and position deivatives
    all_r = []
    all_v = []
    all_drdt = []
    delta_to_v = 3*N
    cm = np.array([0,0,0])
    cm_v = np.array([0,0,0])
    #Turn masses array to flat numpy array for ease
    masses = np.array(masses).flatten()
    tMass = np.sum(masses)
    #Iterate the data set and fill arrays with required values
    for i in range(N):
        all_r.append(rv[3*i:3*(i+1)])
        #v_i = rv[(3*i+delta_to_v):(3*(i+1)+delta_to_v)]
        #all_v.append(v_i)
        all_drdt.append(rv[(3*i+delta_to_v):(3*(i+1)+delta_to_v)]*k2)
        #print(cm,all_r[i]*masses[i])
        cm=np.add(cm,all_drdt[i]*masses[i])
        #cm_v= np.add(cm_v, all_v[i]*masses[i])
        #tMass += masses[i]
    cm/=tMass
    #cm_v/=tMass
    #Convert to numpy arrays for efficiences and ease
    all_r = np.array(all_r)
    all_v = np.array(all_v)
    all_drdt = np.array(all_drdt)

    for i in range(N):
        all_drdt[i] -= cm

    #Create matrix of distances between each body
    rs = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            #Any distance r_ij for j=i is 0
            if(i==j):
                continue
            #rs_ij=rs_ji, prevents double calculations
            if(rs[j,i] != 0):
                rs[i,j] = rs[j,i]
            else:
                #Calculate distance between bodies i and j
                rs[i,j] = np.linalg.norm(all_r[j]-all_r[i])


    #Initiate velocity derivative array
    all_dvdt=[]
    #Iterate each body
    for i in range(N):
        #Initiate the velocity derivative for body i as (0,0,0), to prepair for calcualtion
        body_i_pre_mult = np.zeros(3)
        for j in range(N):
            if(j!=i):
                #Add the acceleration contribution from body j to the body i
                body_i_pre_mult += masses[j]*(all_r[j]-all_r[i])/rs[i,j]**3
        #Add total calcualted velocity change to total array
        all_dvdt.append(k1*body_i_pre_mult)

    #Turn to numpy arrays, concatenate, and flatten, then return to odeint
    all_dvdt = np.array(all_dvdt)
    return np.concatenate((all_drdt, all_dvdt)).flatten()

def PrepairValues(bodies):
    """
    Takes a list of bodies and turns into a set of values to be used by the solver
    Inputs:
        bodies - array like, list Body objects
    Outputs:
        initial - array like, array of initial position and velocity of all bodies
        masses - array like, array of masses of all bodies.
    """
    #Prepair empty lists
    masses = []
    positions = []
    velocities =[]
    #iterate each body and append relevent lists
    for body in bodies:
        masses.append(body.mass)
        positions.append(body.startPos)
        velocities.append(body.startVel)
    #Create array of initial positions and velocities, then flatten
    initial = np.array(positions+velocities)
    initial = initial.flatten()
    #Create array of masses, then return initial values and masses
    masses = np.array(masses)
    return initial, masses


def CreateRogueBody(bodies, radii, mass_scale, dist_scale, vel_scale):
    """
    Creates and returns a new body that fullfils the requirement of the
    body in the rogue body approach value for the system of bodies supplied:
    Inputs:
        bodies: Array like, list of bodies in the system to test
        radii: the ditance (in multiples of the furthest initial state of Any
            object from the systems centre of mass)
        mass_scale: The mass scale used for the solver
        dist_scale: The distance scale used for the solver
        vel_scale: The velocity scale used for the solver
    Out:
        Rb - the rogue body at the specified rogue body distance
    """
    #Calculate total mass of system, and its centre of mass
    tMass = 0
    cMass = np.zeros((3))
    for body in bodies:
        tMass += body.mass
        cMass += np.array(body.startPos)*body.mass
    cMass/=tMass

    #Loop through all bodies to determine which has a furthest initial position
    #from the system centre of mass
    furthest_body = bodies[0]
    furthest_body_dist = np.sum((np.array(bodies[0].startPos)-cMass)**2)

    for i in range(1,len(bodies)):
        rel_pos = np.array(bodies[i].startPos) - cMass
        #Calculate distance of body i to COM
        dist_to_cm = np.sum((rel_pos-cMass)**2)
        #print(furthest_body.name, rel_pos, furthest_body_dist, bodies[i].name, dist_to_cm)
        #Check if its further than last checked object
        if(dist_to_cm > furthest_body_dist):
            #if so, set it as furthest body
            furthest_body=bodies[i]
            furthest_body_dist = dist_to_cm

    #Calculte the distance (in units of dist_scale) of the body from the system
    rogue_radius = np.sqrt(furthest_body_dist) * radii

    #Find system mass and radius in SI units
    SI_mass = tMass*mass_scale #Into KG
    SI_radi = rogue_radius * dist_scale #Into m

    #Calcuate escape velocity
    SI_v_escape = np.sqrt(2*G*SI_mass/SI_radi) #In m/s
    v_escape_scaled = SI_v_escape/vel_scale #In terms of vel_scale

    return Body("Rogue Body", tMass, [rogue_radius,0,0], [0,v_escape_scaled*1.5, 0])


"""
Code Use
NBodyPlotter.py is designed such that the user can easily simulated any n body
system, and either plot or animate.
Example code commented out shows use of code
"""

"""
#Initialises the solver with default values
solver = NBodySolver()

#Add 4 bodies to the solver with iniial starting conditions
solver.AddBody(Body("sun",1e6, [-145000,0,0], [0,-10,0]))
solver.AddBody(Body("second_sun",1e6,[145000, 0, 0], [0,10,0]))
solver.AddBody(Body("third_sun", 1e6, [0,145000,0], [-10,0,0]))
solver.AddBody(Body("third_sun", 1e6, [0,-145000,0], [10,0,0]))
#Define a time span of 5 solar years, with 12000 data points total
time_span=np.linspace(0,5,12000)

#Solver the problem over the time span, and save to "test.png"
solver.SolveNBodyProblem(time_span)
solver.PlotNBodySolution(saveFile="test.png")
"""
