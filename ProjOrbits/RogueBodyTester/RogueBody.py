import sys
sys.path.append('C:\\Users\\nikit\\AppData\\Local\\Programs\\Python\\python38\\lib\\site-packages')

import NBodyPlotter as nbp
from NBodyPlotter import NBodySolver
from NBodyPlotter import Body
import matplotlib.pyplot as plt
import numpy as np

def CreateRogueBody(bodies, radii, mass_scale, dist_scale, vel_scale):
    """
    Creates and returns a new body that fullfils the requirement of the
    body in the rogue body approach value for the system of bodies supplied:
    Inputs:
        bodies: Array like, list of bodies in the system to test
        radii: the ditance (in multiples of the furthest initial state of Any
            object from the systems centre of mass)
    Out:
        Rb - the rogue body
    """
    #Calculate total mass of system, and its centre of mass
    tMass = 0
    cMass = np.zeros((3))
    for body in bodies:
        tMass += body.mass*mass_scale
        cMass += np.array(body.startPos)*dist_scale*body.mass*mass_scale
    cMass/=tMass

    #Loop through all bodies to determine which has a furthest initial position
    #from the system centre of mass
    furthest_body = bodies[0]
    furthest_body_dist = np.sum((np.array(bodies[0].startPos)*dist_scale-cMass)**2)

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
    SI_mass = tMass
    SI_radi = rogue_radius*dist_scale
    SI_v_escape = np.sqrt(2*nbp.G*SI_mass/SI_radi)
    #Calcuate escape velocity
    v_escape_scaled = SI_v_escape/vel_scale
    scaled_mass = SI_mass/mass_scale
    #print(SI_v_escape)
    return Body("Rogue Body", scaled_mass, [rogue_radius,0,0], [0,v_escape_scaled*1.5, 0])
