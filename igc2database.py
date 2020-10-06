
# IGC data to database

class Flight:
    def __init__(self):
        self.igc
        self.date

        self.fix #list
        self.segment #list

        self.windEst
        self.liftTopEst
        self.avgLiftEst
    
    def parse(self):
        kalman_filter =
        while:
            Fix


class Wind:
    def __init__(self):
        self.direction
        self.speed

class Fix:
    def __init__(self):
        self.timestamp
        self.position
        self.altitude
        self.startFilter
        self.endFilter

        self.xEstimate
        self.yEstimate
        self.hEstimate

        self.xWindEstimate
        self.YWindEstimate
        self.liftEstimate

        self.xAirSpeedEstimate
        self.yAirSpeedEstimate
        self.verticalSpeedEstimate

        self.thetaEstimate
        self.liftEstimate


class KalmanFilter:
    def __init__(self,x0, P0, Q0, R0):
        self.m = 500.0
        self.g = 9.81

        self.a = ??
        self.ldMax = 42
        self.minSink = 0.7
        
        # State estimate
        self.x = x0
        # State uncertainty
        self.P = P0
        # Covarience of process noise
        self.G = 
        self.Q = Q0 * [] * dt # sigma_wx = .001
        # Covarience of observation noise
        self.R = R0

        # Observation model
        self.H = constant

    def update(self, z, dt):
        self.x_pre = self.f(dt)
        self.jacob = self.jacobian()
        self.P_pre = self.jacob * self.P * self.jacob.T + self.Q

        
        self.K = self.P_pre * self.H.T * Inv(self.H * self.P_pre self.H.T + self.R)
        self.x = self.x_pre + self.K * (z - self.h())
        self.P = (I - self.K * self.H) * self.P_pre 

    def f(self, dt):
        dt =

        # Aux variables
        v_post = sqrt(v_x_post ** 2 + v_y_post ** 2)
        drag_post = (s_min + a / 2 (v_post - v_minsink) ** 2) / v_post *  lift_post
        
        a_long_post = (cos(theta) * - (h_dot_post - wind_h_post) / v_post * lift_post - drag_post) / m
        a_lat_post = sin(theta) * lift_pre / m

        v_x_dot = v_x_post / v_post + a_long_post + v_y_post / v_post + a_lat_post
        v_y_dot = v_y_post / v_post + a_long_post + v_x_post / v_post + a_lat_post
        
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
        v_y_pre = v_y_post + v_y_dot *dt # Process noise by theta, lift, wind_h, v_x, v_y 

        theta_pre = theta_post # Process noise by theta
        lift_pre = lif_post # Process noise by lift

    def jacobian(self):
        self.jacobian[:] = I

        


    def h(self):

class Position:
    def __init__(self):
        self.latWGS
        self.lonWGS
        self.latUTM
        self.lonUTM
        self.projection

class LiftArea:        
    def __init__(self):
        self.position #list
        self.radius
        self.landCover #list
        self.soilType #list
        self.minAltitude
        self.maxAltitude

        self.altitudeDifference

class Projection: 
    def __init__(self):
        self.ellps
        self.zone
        self.proj
        self.func


class Database:
    def __init__(self):

def main():
    pass
    #Create database

    #Open IGC files and save fixes to db

# Parse arguments
if __name__ == '__main__':
    igc_folder=os.getcwd()+"/EFRYarkisto"

