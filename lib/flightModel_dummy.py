from math import sqrt, acos, asin, cos, sin
import numpy

class Model:
    def __init__(self):
        self.g = 9.81 # m/s^2
        self.k0 = 1.2
        self.k1 = 1.3

        self.stateVariables = ('Latitude','Longtitude','Altitude', 'Wind_x','Wind_y','Wind_z', 'GS_x','GS_y', 'GS_z','Theta','n')
    
    # Intial state
    def getX0(self, initialFix):
        return numpy.transpose(numpy.array([[initialFix.latitude, initialFix.longitude, initialFix.altitude, 0, 0, 0, 0, 0, 0, 0, 1]]))
    # State uncertainty
    def getP0(self):
        return numpy.array([5,5,5,0,0,0,5,5,5,0,0])
        
    # Covarience of process noise
    def getQ0(self, dt):
        G = numpy.zeros((11,5))
        G[0,0] = (dt**2) / 2
        G[1,1] = (dt**2) / 2
        G[2,2] = (dt**2) / 2
        G[7,0] = dt
        G[6,1] = dt
        G[8,2] = dt
        Q = numpy.identity(5)*numpy.array([.1,.1,.05,0,0])
        return numpy.matmul(numpy.matmul(G,Q),G.T)

    # Covarience of observation noise
    def getR(self):
        r = 5
        return numpy.array([[r,0,0],[0,r,0],[0,0,r*2]])

    # Observation model
    def getH(self, dt):
        dt = 0
        return numpy.array([[1,0,0,0,0,0,0,dt,0,0,0],[0,1,0,0,0,0,dt,0,0,0,0],[0,0,1,0,0,0,0,0,dt,0,0]])

    def Predict(self, x, dt):

        # State variables
        xOut = x
        xOut[6,0] = x[6,0] 
        xOut[7,0] = x[7,0] 
        xOut[8,0] = x[8,0] 
        xOut[0,0] = x[0,0] + x[7,0] * dt
        xOut[1,0] = x[1,0] + x[6,0] * dt
        xOut[2,0] = x[2,0] + x[8,0] * dt
        
        return xOut
    
    def Jacobian(self, x, dt):
        F = numpy.identity(11)
        F[0,7]=dt
        F[1,6]=dt
        F[2,8]=dt

        return F
