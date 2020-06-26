def SolveNBodyProblem(bodies, k1, k2, time_span):
    """
    Solves the N body problem for the given bodies, based on the supplied constants
    over the given time span.
    Inputs:
        bodies - list of all bodies to calculate solution for
        k1, k2 - constants calculated using SetDefaultReatives or SetRelativeValues
        time_span - the time span relative to the time scale the solution should
            be found over
    Outputs:
        bodySol - The positions the solution of all bodies
        n_body_sol - The initial calculated solution, contains position and velocity
            of all bodies.
    """
    #Prepair values based on list of bodies
    initial, masses = PrepairValues(bodies)
    N = len(bodies)
    #solve for all bodies
    n_body_sol = integrate.odeint(CoupledNBodyODE, initial, time_span, args=(k1,k2,N, [masses]))
    #Create array of just the positions of the solution
    bodySol = []
    for i in range(N):
        bodySol.append(n_body_sol[:,(3*i):(3*(i+1))])
    #Return both the neat solution, as well as the full solution
    return bodySol, n_body_sol




def PlotNBodySolution(bodies, bodySol, N, ax=None, saveFile=None):
    """
    Plots each of the N solutions.
    Inputs:
        bodySol - The position solution of all bodies
        N - The number of bodies
        ax - The axis to plot on, if none supplied, creates one.
    """
    #Check if an axis to plot on has been supplied, if not, create a new one
    if(ax==None):
        fig=plt.figure(figsize=(5,5))
        ax=fig.add_subplot(111,projection="3d")
    #Iterate each body and extract full path for each body, then plot.
    for i in range(N):
        b = bodySol[i]
        ax.plot(b[:,0],b[:,1],b[:,2],color=colours[i], label=bodies[i].name)
    #Add details to plot, then show
    ax.set_xlabel("x-coordinate",fontsize=14)
    ax.set_ylabel("y-coordinate",fontsize=14)
    ax.set_zlabel("z-coordinate",fontsize=14)
    ax.set_title("Visualization of a 3 body system\n",fontsize=14)
    ax.legend(loc="upper left",fontsize=14)
    plt.show()
    #Check if a save file is specified, if so, save plot
    if(saveFile != None):
        fig.savefig(saveFile, dpi=fig.dpi)

def AnimateNBodyProblem(bodySol, N, axis_size=None):
    """
    Plots and animates each of the N solutions.
    Inputs:
        bodySol - The position solution of all bodies
        N - The number of bodies
        axis_size - The size of the axis, if none is supplied, the maximum
            position out of all the solutions is used
    """
    #Create empty array, then fill with extracted data of each body
    data = []
    for i in range(N):
        data.append([bodySol[i][:,0],bodySol[i][:,1],bodySol[i][:,2]])
    #Turn to numpy array
    data = np.array(data)
    #Check if axis_size is defined. If not, define it
    if(axis_size==None):
        axis_size = np.max(bodySol)
    #Create 3d figure to plot on
    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot(111,projection="3d")
    #Extract data into a set of 3 dimensional lines
    lines = [ax.plot(dat[0,0:1],dat[1,0:1],dat[2,0:1])[0] for dat in data]

    def update_lines(num, dataLines, lines):
        """
        Update function for the animation.
        Inputs:
            num - the current iteration of the animation
            dataLines - all of the 3d position solutions of all bodies
            lines - the lines to animate
        """
        for line, data in zip(lines, dataLines):
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
        return lines
    #Set up axis of plot
    ax.set_xlim3d([-axis_size,axis_size])
    ax.set_xlabel('X')

    ax.set_ylim3d([-axis_size,axis_size])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-axis_size,axis_size])
    ax.set_zlabel('Z')
    #Create animation, then show to user
    line_ani = FuncAnimation(fig, update_lines, len(time_span), fargs=(data, lines),
                                   interval=0.1, blit=True, repeat=False)
    plt.show()
