#from lib.database import Database
#from lib.flightModel_dummy_2 import Model
#from math import sqrt, acos, cos, sin
from numpy import zeros, matmul, array 

from pandas import DataFrame 
#import matplotlib.pyplot as plt
from sqlite3 import connect
#from time import strftime, gmtime



class Flight:
    def __init__(self, flightID, database):
        self.flightID = flightID
        database.cursor.execute('SELECT Timestamp, Latitude, Longitude, Altitude FROM Fixes WHERE FlightID=? ORDER BY Timestamp ASC',(self.flightID,))
        self.Fixes = pandas.DataFrame(database.cursor.fetchall(), columns=['Timestamp', 'Latitude', 'Longitude', 'Altitude'])

    def estimate(self):
        # Create Estimator  and calculate estimates
        #currentFilter = KalmanFilter(self.Fixes[0])
        currentFilter = StateObserver(self.Fixes[0:1].to_numpy())
        #Estimate
        for row in self.Fixes:
            print(row)
            #currentEstimate = currentFilter.update(fix)
            #fix.setEstimate(currentEstimate)
    
    def getEstimateData(self):
        output = list()
        for fix in self.Fixes:
            output.append((self.flightID, fix.timestamp, fix.estLatitude, fix.estLongitude, fix.estAltitude, fix.xwind, fix.ywind, fix.zwind, fix.speed, fix.direction,fix.verticalSpeed, fix.theta, fix.n))
        return output

    def plot(self):
        n = len(self.Fixes)
        data = numpy.zeros((n,10))
        t0=self.Fixes[0].timestamp
        GS_x_old = 0
        GS_y_old = 0
        GS_z_old = 0
        time_old = t0 - 1

        for i in range(n):
            fix=self.Fixes[i]
            dt = (fix.timestamp-time_old)
            g = sqrt((fix.GS_x-GS_x_old)**2+(fix.GS_y-GS_y_old)**2+(fix.verticalSpeed-GS_z_old+9.81*dt)**2)/ dt / 9.81
            data[i,:]=(fix.timestamp-t0,sqrt( (fix.estLatitude-fix.latitude)**2 +(fix.estLongitude-fix.longitude)**2),fix.estAltitude-fix.altitude, sqrt((fix.xwind**2)+( fix.ywind**2)), fix.zwind, fix.speed,fix.verticalSpeed, fix.n, g, fix.theta)
            GS_x_old = fix.GS_x
            GS_y_old = fix.GS_y
            GS_z_old = fix.verticalSpeed

            time_old = fix.timestamp
        fig, ((ax0,ax1),(ax2, ax3),(ax4,ax5), (ax6, ax7),(ax8,ax9)) = plt.subplots(nrows=5,ncols=2)

        ax0.plot(data[:,0],data[:,1])
        ax0.set_ylabel('Lateral error')
        ax0.grid(True)

        ax1.plot(data[:,0],data[:,2])
        ax1.set_ylabel('Vertical error')
        ax1.grid(True)

        ax2.plot(data[:,0],data[:,3])
        ax2.set_ylabel('Wind speed')
        ax2.grid(True)

        ax3.plot(data[:,0],data[:,4])
        ax3.set_ylabel('Lift')
        ax3.grid(True)

        ax4.plot(data[:,0],data[:,5])
        ax4.set_ylabel('Speed')
        ax4.grid(True)

        ax5.plot(data[:,0],data[:,6])
        ax5.set_ylabel('Vertical speed')
        ax5.grid(True)

        ax6.plot(data[:,0],data[:,7])
        ax6.set_ylabel('n')
        ax6.grid(True)

        ax7.plot(data[:,0],data[:,8])
        ax7.set_ylabel('G')
        ax7.grid(True)

        ax8.plot(data[:,0],data[:,9])
        ax8.set_ylabel('theta')
        ax8.grid(True)
        
        #ax8.plot(data[:,0],data[:,10])
        #ax8.set_ylabel('G')
        #ax8.grid(True)
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
        self.GS_y = data[6,0]
        self.GS_x = data[7,0]
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
        self.theta = 0#data[9,0]
        self.n = 0#data[10,0]


class StateObserver():
    def __init__(self, initialFix):
        #x: Latitude, longitude, altitude, w_y, w_x, w_z, v_y, v_x, v_z, a_y, a_x, a_z, k0, k1 
        self.x = initialFix
        self.w = numpy.zeros(3)
        self.v = numpy.zeros(3)
        self.a = numpy.zeros(3)
        self.k = 1.2
        self.K = 0.1

    def Update(self, fix):
        # Luenberger observer for x, v and a
        
        # Update wind

        # Update k0 and k1
        return self.x

    def EstimateWind(self, a, TAS, k):
        # TAS_a and TAS_n

        # Solve w_a: equation: Drag from geometry = Drag from formula  
        
        # Divide w_a to x,y,z components
        w=0

        return w


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
        elif 0:#step > 1000*self.dt:
            self.x = self.model.getX0(fix) 
            self.P = self.model.getP0()
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

            return self.x

def main(databaseName):
    # Open database
    conn = connect(databaseName)
    cursor = conn.cursor()
    
    # Clear database table
    cursor.execute("DROP TABLE IF EXISTS Estimates")

    # Get flights    
    cursor.execute('SELECT FlightID FROM Flights', ())
    flightList = cursor.fetchall()
    print(len(flightList)," flights found")

    # Go through flights
    count = 0
    for flight in flightList:
        # Read data from database 
        cursor.execute('SELECT * FROM Fixes WHERE FlightID=?', (flight[0],))
        df = DataFrame(cursor.fetchall(), columns=['FlightID', 'Time', 'Latitude','Longitude','Altitude'])
        
        # Calculate dt
        df['dt'] = df['Time'].diff()

        # Estimate speed and acceleration
        #TODO: df.add(Minmax_altitude)
        df.add(EstimateAcceleration(df, 'Latitude'), columns = ['v_y','a_y']]
        df.add(EstimateAcceleration(df, 'Longitude'), columns = ['v_x','a_x']]
        df.add(EstimateAcceleration(df, 'Altitude'), columns = ['v_z','a_z']]

        # Estimate wind
        df.add(EstimateWind(df), columns = ['w_y','w_x','w_z']]

        # Save to database
        #df.plot()
        df.to_sql('Estimates', conn, if_exists='append', index = False)

        # Print status
        count += 1
        print("\r",count," flight processed", end='')

    # Close database
    conn.close()
    print("\nCompleted")

def EstimateAcceleration(dataframe, key):
    n = len(dataframe.index) 
    dx = dataframe[key].diff()
        
    # Initialize matrixes    
    K = eye((2,2))
    q = zeros((2,1))

    #K_ = eye((2,2))
    #q_ = zeros((2,1))
    
    
    # Factors for recursion
    for i in range(1,n):
        dt = dataframe['dt'][i]

        A = array([[],[]])
        b = array([[],[]])
        K = matmul(A, K)
        q = matmul(A, q)) + b
        
        #dt_ = dataframe['dt'][-i]
        #A_ = array([[],[]])
        #b_ = array([[],[]])
        #K_ = matmul(A_, K_)
        #q_ = matmul(A_, K_) + b_
    
    # Solve first point
    x0=array([q[-1,1]/K[-1, 1, 0], 0])

    # Calculate other points
    result = matmul(K, x[0:2]) + q 

    return result

def EstimateWind(dataframe):
    
    TAS = 0
    A = 0 
    
    # TAS_a and TAS_n
    TAS_lift = numpy.dot(TAS, A)
    TAS_drag = TAS - TAS_a
    
    # Solve w_a: equation: Drag from geometry = Drag from formula  
        
    # Divide w_a to x,y,z components
    dw = 0
    w = numpy.cumsum(dw)
    return w

if __name__ == '__main__':
    
    databaseName = "LiftData_20201017_171445"
    #parameters = {saveFilterStates = True, recalculateAll = True}
    main(databaseName)