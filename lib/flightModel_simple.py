from math import sqrt, acos, cos, sin
import numpy

class Model:
    def __init__(self, initialFix):
        self.m = 500.0 # kg
        self.g = 9.81 # m/s^2
        self.minSink = 0.7 # m/s
        self.minSinkSpeed = 24.0 # m/s
        self.highSpeed = 56.0 # 
        self.sinkForHighSpeed = 2.0 # m/s
        self.a = numpy.dot(2,(self.sinkForHighSpeed - self.minSink)) / (self.highSpeed - self.minSinkSpeed) ** 2
        


        self.x0 = numpy.transpose(numpy.array([[initialFix.latitude, initialFix.longitude, initialFix.altitude, 0, 0, 0, 0, 0, 0, self.m * self.g, 0]]))
        self.x_pre = self.x0

    # State uncertainty
    def getP0(self):
        return numpy.array([5,5,5,5,5,5,5,5,5,0.1,1])
        
    # Covarience of process noise
    def getQ0(self):
        return numpy.array([0,0,0,0.1,0.1,0.2,0.5,0.5,0.5,0.2,0.2])*numpy.identity(11)

    # Covarience of observation noise
    def getR(self):
        return numpy.array([[5,0,0],[0,5,0],[0,0,5]])

    # Observation model
    def getH(self):
        return numpy.array([[1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0]])

    def updatePrediction(self, x, dt):

        # Aux variables
        v = sqrt(x[6,0] ** 2 + x[7,0] ** 2)
        a_z = cos(x[9,0]) * x[10,0] / self.m - self.g
        a_lat = sin(x[9,0]) * x[10,0] / self.m

        if v == 0: 
            drag = 0
            a_lon = 0
            a_x = 0
            a_y = 0
        else:
            drag = (self.minSink + self.a / 2 * (v - self.minSinkSpeed) ** 2) / v * x[10,0]
            a_lon = (cos(x[9,0]) * - (x[8,0] - x[5,0]) / v * x[10,0] - drag) / self.m
            a_x = x[6,0] / v * a_lon + x[7,0] / v * a_lat
            a_y = x[7,0] / v * a_lon + x[6,0] / v * a_lat
        
        # State variables
        self.x_pre[1,0] = x[1,0] + x[6,0] * dt + x[3,0] * dt + 1 / 2 * a_x * dt ** 2 # Process noise by wind_x
        self.x_pre[0,0] = x[0,0] + x[7,0] * dt + x[4,0] * dt + 1 / 2 * a_y * dt ** 2 # Process noise by wind_y
        self.x_pre[2,0] = x[2,0] + x[8,0] * dt + 1 / 2 * a_z * dt ** 2 # Process noise by theta, lift
        self.x_pre[8,0] = x[8,0] + a_z * dt # Process noise by theta, lift

        self.x_pre[3,0] = x[3,0] # Process noise by wind_x
        self.x_pre[4,0] = x[4,0] # Process noise by wind_y
        self.x_pre[5,0] = x[5,0] # Process noise by wind_h

        self.x_pre[6,0] = x[6,0] + a_x * dt # Process noise by theta, lift, wind_h, v_x, v_y 
        self.x_pre[7,0] = x[7,0] + a_y * dt # Process noise by theta, lift, wind_h, v_x, v_y 

        self.x_pre[9,0] = x[9,0] # Process noise by theta
        self.x_pre[10,0] = x[10,0] # Process noise by lift

        return self.x_pre
    
    def updateJacobian(self, x, dt):
        F = numpy.identity(11)

        # Latitude
        F[0,2] = dt
        F[0,2] = dt

        # Longitude
        F[1,2] = dt
        F[1,2] = dt

        # Altitude

        # Vx

        # Vy

        # Vz

        return F

    def f(self):

        # Aux variables
        v_post = sqrt(v_x_post ** 2 + v_y_post ** 2)
        drag_post = (s_min + a / 2 (v_post - v_minsink) ** 2) / v_post *  lift_post
        
        a_long_post = (cos(theta) * - (h_dot_post - wind_h_post) / v_post * lift_post - drag_post) / m
        a_lat_post = sin(theta) * lift_pre / m

        v_x_dot = v_x_post / v_post * a_long_post + v_y_post / v_post * a_lat_post
        v_y_dot = v_y_post / v_post * a_long_post + v_x_post / v_post * a_lat_post
        
        h_dot_dot = cos(theta) * lift_post / m - g

        # State variables
        x_pre = x_post + v_x_post * dt + wind_x_post * dt + 1 / 2 * v_x_dot * dt ** 2 # Process noise by wind_x
        y_pre = y_post + v_y_post * dt + wind_y_post * dt + 1 / 2 * v_y_dot * dt ** 2 # Process noise by wind_y
        h_pre = h_post + h_dot_post * dt + 1 / 2 * h_dot_dot * dt ** 2 # Process noise by theta, lift
        h_dot_pre = h_dot_post + h_dot_dot * dt # Process noise by theta, lift

        wind_x_pre = wind_x_post # Process noise by wind_x
        wind_y_pre = wind_y_post # Process noise by wind_y
        wind_h_pre = wind_h_post # Process noise by wind_h

        v_x_pre = v_x_post + v_x_dot * dt # Process noise by theta, lift, wind_h, v_x, v_y 
        v_y_pre = v_y_post + v_y_dot * dt # Process noise by theta, lift, wind_h, v_x, v_y 

        theta_pre = theta_post # Process noise by theta
        lift_pre = lif_post # Process noise by lift