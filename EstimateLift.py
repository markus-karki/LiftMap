from lib.database import Database
from lib.flightModel import Model
from math import sqrt, acos, cos, sin
import numpy
import matplotlib.pyplot as plt


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

    def plot(self):
        n = len(self.Fixes)
        data = numpy.zeros((n,12))
        t0=self.Fixes[0].timestamp
        GS_x_old = 0
        GS_y_old = 0
        GS_z_old = 0
        time_old = t0 - 1

        for i in range(n):
            fix=self.Fixes[i]
            dt = (fix.timestamp-time_old)
            g = sqrt((fix.GS_x-GS_x_old)**2+(fix.GS_y-GS_y_old)**2+(fix.verticalSpeed-GS_z_old+9.81*dt)**2)/ dt / 9.81
            data[i,:]=(fix.timestamp-t0, fix.estLatitude-fix.latitude, fix.estLongitude-fix.longitude, fix.estAltitude-fix.altitude, fix.xwind, fix.ywind, fix.zwind, fix.speed, g,fix.verticalSpeed, fix.theta, fix.n)
            GS_x_old = fix.GS_x
            GS_y_old = fix.GS_y
            GS_z_old = fix.verticalSpeed

            time_old = fix.timestamp
        fig, ((ax0,ax1,ax2),(ax3,ax4,ax5)) = plt.subplots(nrows=2,ncols=3)

        ax0.plot(data[:,0],data[:,1:3])
        ax0.set_ylabel('Position error')
        ax0.grid(True)

        ax1.plot(data[:,0],data[:,4:6])
        ax1.set_ylabel('Wind')
        ax1.grid(True)

        ax2.plot(data[:,0],data[:,7])
        ax2.set_ylabel('Speed')
        ax2.grid(True)

        ax3.plot(data[:,0],data[:,9])
        ax3.set_ylabel('Vertical speed')
        ax3.grid(True)

        ax4.plot(data[:,0],data[:,10:11])
        ax4.set_ylabel('theta, n')
        ax4.grid(True)

        ax5.plot(data[:,0],data[:,8])
        ax5.set_ylabel('G')
        ax5.grid(True)
        
        #fig.suptitle("At finish line",y=0.99)

        plt.tight_layout()
        plt.show()
        #Paikkavirhe x,y,z

        #Wind mag, dir

        #Lift, vertical speed

        #Speed mag, turnrate
        
        #theta, n

    
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
        self.GS_x = data[6,0]
        self.GS_y = data[7,0]
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

        self.time = initialFix.timestamp

        # State uncertainty
        self.P = self.model.getP0()

        # Covarience of observation noise
        self.R = self.model.getR()


    def update(self, fix):
        # Update time
        self.dt = fix.timestamp - self.time
        self.time = fix.timestamp
        self.H = self.model.getH(self.dt)

        step = sqrt((fix.latitude-self.x[0,0])**2+(fix.longitude-self.x[1,0])**2+(fix.altitude-self.x[2,0])**2)

        if self.dt == 0:
            return self.x
        elif step > 100*self.dt:
            self.x[0,0]=fix.latitude
            self.x[1,0]=fix.longitude
            self.x[2,0]=fix.altitude

            return self.x
        else:

            # Predict
            self.x = self.model.Predict(self.x, self.dt)
            self.F = self.model.Jacobian(self.x, self.dt)
            self.P = self.F * self.P * self.F.T + self.model.getQ0(self.dt)
            
            # Update
            self.y = numpy.array([[fix.latitude,fix.longitude,fix.altitude]]).T - numpy.matmul(self.H, self.x)
            self.S = numpy.matmul(numpy.matmul(self.H, self.P),self.H.T) + self.R
            self.K = numpy.matmul(numpy.matmul(self.P, self.H.T), numpy.linalg.pinv(self.S))
            self.x = self.x + numpy.matmul(self.K, self.y)
            self.P = numpy.matmul((numpy.identity(len(self.x)) - numpy.matmul(self.K, self.H)), self.P) 
            print(self.x[7,0])
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
        currentFlight.plot()
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