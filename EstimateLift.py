from lib.database import Database
from lib.flightModel import Model
from math import sqrt, acos, cos, sin
import numpy

class Flight:
    def __init__(self, flightID, database):
        self.flightID = flightID
        self.Fixes =  list()  
        self.Estimates =  list()
        self.FilterStates =  list() 
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
            output.append((self.flightID, fix.timestamp, fix.estLatitude, fix.estLongitude, fix.estAltitude, fix.xwind, fix.ywind, fix.zwind, fix.speed, fix.direction,fix.verticalSpeed, fix.theta, fix.n))
        return output
    
    def getStateVariableData(self):
        output = list()
        for fix in self.Fixes:
            output.append((self.flightID, fix.timestamp, fix.variable, fix.value))
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

        #for dataPoint in data[6:,0]
        self.speed = sqrt(data[6,0]**2 + data[7,0] ** 2)

        if self.speed == 0:
            self.direction = 0
        else:
            direction = acos(data[7,0] / self.speed) * 57.296
            if data[6]<0:
                self.direction = 360 - direction
            else:
                self.direction = direction

        self.verticalSpeed = data[8,0]
        self.theta = data[9,0]
        self.n = data[10,0]

class KalmanFilter:
    def __init__(self, initialFix):
        self.model = Model()        
        
        # Inital state
        self.x = self.model.getX0(initialFix) 

        
        self.x_pre = self.x
        self.time = initialFix.timestamp

        # State uncertainty
        self.P = self.model.getP0()

        # Covarience of observation noise
        self.R = self.model.getR()

        # Observation model
        self.H = self.model.getH()

    def update(self, fix):
        # Update time
        self.dt = fix.timestamp - self.time
        self.time = fix.timestamp
        
        # Covarience of process noise
        self.Q = self.model.getQ0() * self.dt

        if self.dt == 0:
            return self.x
        else:

            # Predict
            self.x_pre = self.model.Predict(self.x, self.dt)
            self.F = self.model.Jacobian(self.x, self.dt)
            self.P_pre = self.F * self.P * self.F.T + self.Q
            
            # Update
            self.y = numpy.array([[fix.latitude,fix.longitude,fix.altitude]]).T - numpy.matmul(self.H, self.x_pre)
            self.S = numpy.matmul(numpy.matmul(self.H, self.P_pre),self.H.T) + self.R
            self.K = numpy.matmul(numpy.matmul(self.P_pre, self.H.T), numpy.linalg.inv(self.S))
            self.x = self.x_pre + numpy.matmul(self.K, self.y)
            self.P = numpy.matmul((numpy.identity(len(self.x)) - numpy.matmul(self.K, self.H)), self.P_pre) 

            return self.x

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
    
    databaseName = "LiftData_20201014_163401"
    #parameters = {saveFilterStates = True, recalculateAll = True}
    main(databaseName)