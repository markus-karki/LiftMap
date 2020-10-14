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
        return numpy.array([5,5,5,5,5,5,5,5,5,0.1,1])
        
    # Covarience of process noise
    def getQ0(self):
        a=numpy.array([[1,1,1,0.1,0.1,0.2,0.5,0.5,0.5,0.2,0.2]])
        return a*numpy.ones((11,11))*a.T

    # Covarience of observation noise
    def getR(self):
        return numpy.array([[5,0,0],[0,5,0],[0,0,5]])

    # Observation model
    def getH(self):
        return numpy.array([[1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0]])

    def Predict(self, x, dt):

        TAS_x = x[6,0] - x[3,0] # GS_x - Wind_x
        TAS_y = x[7,0] - x[4,0] # GS_y - Wind_y
        TAS_z = x[8,0] - x[5,0] # GS_z - Wind_z
        TAS_lat = sqrt(TAS_x ** 2 + TAS_y ** 2)
        TAS = sqrt(TAS_x ** 2 + TAS_y ** 2 + TAS_z ** 2)
        
        
        if (TAS_lat < 15 or x[2,0]<200):
            a_x = 0
            a_y = 0 
            a_z = 0

        else:  
            a_lift = x[10,0] * self.g
            a_drag = self.k0 / 10000 * (TAS ** 2)  + self.k1 * 100 * (x[10,0] ** 2)/(TAS ** 2) 

            cos_gamma = TAS_lat / TAS
            sin_gamma = TAS_z / TAS
            cos_beta = TAS_y / TAS_lat
            sin_beta = TAS_x / TAS_lat
            cos_theta = cos(x[9,0])
            sin_theta = sin(x[9,0])

            a_x = (cos_beta * sin_gamma * sin_theta - sin_beta * cos_theta) * a_lift - sin_beta * cos_gamma * a_drag 
            a_y = (- sin_beta * sin_gamma * sin_theta - cos_beta * cos_theta) * a_lift - cos_beta * cos_gamma * a_drag 
            a_z = cos_gamma * cos_theta * a_lift - sin_gamma * a_drag  - self.g

        # State variables
        xOut = x
        xOut[6,0] = x[6,0] + a_x * dt
        xOut[7,0] = x[7,0] + a_y * dt
        xOut[8,0] = x[8,0] + a_z * dt
        xOut[0,0] = x[0,0] + x[7,0] * dt + a_y / 2 * (dt ** 2)
        xOut[1,0] = x[1,0] + x[6,0] * dt + a_x / 2 * (dt ** 2)
        xOut[2,0] = x[2,0] + x[8,0] * dt + a_z / 2 * (dt ** 2)
        
        return xOut
    
    def Jacobian(self, x, dt):
        F = numpy.identity(11)
        if (x[2,0]<200):
            F[0,7]=dt
            F[1,6]=dt
            F[2,8]=dt
        else:
            rows = (0,1,2,6,7,8)
            cols = (3,4,5,6,7,8,9,10)
            Q = self.getQ0()

            for i in rows:
                for j in cols:
                    dx = numpy.zeros([11,1])
                    dx[j] = Q[j, j]
                    F[i,j] = (self.Predict(x+dx, dt)[i] - self.Predict(x-dx, dt)[i]) / (2 * Q[j, j])
        
        return F