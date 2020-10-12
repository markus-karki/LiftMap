from lib.database import Database
from math import sqrt, acos
import numpy

class Flight:
    def __init__(self, flightID, database):
        self.flightID = flightID
        self.Fixes =  list()   
        data = database.getData('SELECT Timestamp, Latitude, Longitude, Altitude FROM Fixes WHERE FlightID=? ORDER BY Timestamp ASC',(self.flightID,))
        for row in data:
            newFix = Fix(row)
            self.Fixes.append(newFix)

    def estimate(self, currentFilter):
        for fix in self.Fixes:
            currentEstimate = currentFilter.update(fix)
            fix.setEstimate(currentEstimate)
    
    def getEstimateData(self):
        output = list()
        for fix in self.Fixes:
            output.append((self.flightID, fix.timestamp, fix.estLatitude, fix.estLongitude, fix.estAltitude, fix.xwind, fix.ywind, fix.zwind, fix.speed, fix.direction, fix.verticalSpeed, fix.theta, fix.lift))
        return output

class Fix:
    def __init__(self, data):
        self.timestamp = data[0]
        self.latitude = data[1]
        self.longitude = data[2]
        self.altitude = data[3]

    def setEstimate(self, data):
        self.estLatitude = data[0,0]
        self.estLongitude = data[1,0]
        self.estAltitude = data[2,0]
        self.xwind = data[3,0]
        self.ywind = data[4,0]
        self.zwind = data[5,0]
        self.speed = int(sqrt(data[6,0]**2 + data[7,0] ** 2))

        if self.speed == 0:
            self.direction = 0
        else:
            direction = acos(data[7,0] / self.speed) * 57.296
            if data[6]<0:
                self.direction = int(360 - direction)
            else:
                self.direction = int(direction)

        self.verticalSpeed = data[8,0]
        self.theta = data[9,0]
        self.lift = data[10,0]

class KalmanFilter:
    def __init__(self, initialFix):
        self.m = 500.0 # kg
        self.g = 9.81 # m/s^2
        self.minSink = 0.7 # m/s
        self.minSinkSpeed = 24.0 # m/s
        self.highSpeed = 56.0 # 
        self.sinkForHighSpeed = 2.0 # m/s


        self.a = numpy.dot(2,(self.sinkForHighSpeed - self.minSink)) / (self.highSpeed - self.minSinkSpeed) ** 2

        # State estimate: (lat, lon, alt, windX, windY, windZ, vX, vY, vZ, theta, lift)
        self.x = numpy.transpose(numpy.array([[initialFix.latitude, initialFix.longitude, initialFix.altitude, 0, 0, 0, 0, 0, 0, 0, 0]]))
        self.time = initialFix.timestamp

        # State uncertainty
        self.P = numpy.array([5,5,5,5,5,5,5,5,5,0.1,1])
        
        # Covarience of process noise
        self.Q0 = numpy.array([0,0,0,0.1,0.1,0.2,0.5,0.5,0.5,0.2,0.2])*numpy.identity(11)

        # Covarience of observation noise
        self.R = numpy.array([[5,0,0],[0,5,0],[0,0,5]])

        # Observation model
        self.H = numpy.array([[1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0]])

    def update(self, fix):
        # Update time
        self.dt = fix.timestamp - self.time
        self.time = fix.timestamp
        
        # Covarience of process noise
        self.Q = self.Q0 * self.dt

        if self.dt == 0:
            return self.x
        else:

            # Predict
            self.updatePredictionDummy()
            self.updateJacobianDummy()
            self.P_pre = self.F * self.P * self.F.T + self.Q
            
            # Update
            self.y = numpy.array([[fix.latitude,fix.longitude,fix.altitude]]).T - numpy.matmul(self.H, self.x_pre)
            self.S = numpy.matmul(numpy.matmul(self.H, self.P_pre),self.H.T) + self.R
            self.K = numpy.matmul(numpy.matmul(self.P_pre, self.H.T), numpy.linalg.inv(self.S))
            self.x = self.x_pre + numpy.matmul(self.K, self.y)
            self.P = numpy.matmul((numpy.identity(11) - numpy.matmul(self.K, self.H)), self.P_pre) 

            return self.x


    def updatePredictionDummy(self):
        self.x_pre = self.x

    def updatePrediction(self):

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

    def f(self):

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

    def updateJacobianDummy(self):
        self.F = numpy.identity(11)

    def updateJacobian(self):
        self.F = numpy.identity(11)


def main(databaseName):
    # Open database
    database = Database(databaseName)
    database.clearEstimateTables()

    # Get flights
    flightList = database.getData('SELECT FlightID FROM Flights', ())
    print(len(flightList)," flights found")
    count = 0

    # Go through flights
    for flight in flightList:
        # Create flight
        currentFlight = Flight(flight[0],database)

        # Create Kalman filter and calculate estimates
        currentFilter = KalmanFilter(currentFlight.Fixes[0])
        currentFlight.estimate(currentFilter)

        # Save to database
        database.insertManyData('INSERT INTO Estimates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',currentFlight.getEstimateData())
        
        # Print status
        count += 1
        print("\r",count," flight processed", end='')

    # Close database
    database.file.close()
    print("\nCompleted")

if __name__ == '__main__':
    
    databaseName = "LiftData_20201012_073620"

    main(databaseName)