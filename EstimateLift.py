from numpy import zeros, matmul, array, empty, roots, real, isreal, sqrt, dot, sum, abs
from scipy.interpolate import UnivariateSpline
from pandas import DataFrame 
import matplotlib.pyplot as plt
from sqlite3 import connect
# TODO 
# - roots -> scipy.optimize.singe_f tms.
# - missing rows

class Sets:
    def __init__(self):
        pass

    def updateStatus(self):
        pass

    def printStatus(self):
        pass

class Dataset:
    def __init__(self, flight):
        cursor.execute('SELECT * FROM Fixes WHERE FlightID=?', (flight[0],))
        self.df = DataFrame(cursor.fetchall(), columns=['FlightID', 'Time', 'y','x','z'])
        self.polar = (1, 1) 

    def estimatateAcceleration(self):
        pass

    def calculateMinmaxAltitude(self):
        pass

    def estimatePolar(self):
        pass

    def estimateWind(self):
        pass

    def estimateWind(self):
        pass
    
    def estimateIfLift(self):
        pass
    
    def estimateLiftLimits(self):
        pass

    def estimateLiftLocations(self):
        pass
    
    def smooth(self): # Smooth, if needed (Kalman filter)
        pass 

    def changeTimeInterval(self):
        pass

    def takeSample(self):
        pass

    def calculateWeights(self):
        pass
    
    def plot(self):
        self.df.plot(x='Time',subplots=1, layout=(5,3), sharex=1, grid=1)
        plt.show()
        self.df.hist(column='w_z', grid=1, bins=100)
        plt.show()

    def saveToDB(self):
        self.df.to_sql('Estimates', conn, if_exists='append', index = False)

def main(databaseName, RecalculateAll):
    # Open database
    conn = connect(databaseName)
    cursor = conn.cursor()
    
    # Clear database table, if needed
    if RecalculateAll:
        cursor.execute("DROP TABLE IF EXISTS Estimates")

    # Get flights    
    cursor.execute('SELECT FlightID FROM Flights', ())
    flightList = cursor.fetchall()
    print(len(flightList)," flights found")

    # Go through flights
    count = 0
    for flight in flightList:
        # Read data from database 
        dataset = Dataset(flight)
        
        # Estimate speed and acceleration
        dataset.estimateAcceleration('y', ['v_y','a_y'])
        dataset.estimateAcceleration('x', ['v_x','a_x'])
        dataset.estimateAcceleration('z',  ['v_z','a_z'], 9.81)
    
        # Smooth, if needed (Kalman filter)

        # Convert to standard time interval, if selected
        
        # Estimate glider polar

        # Estimate wind
        dataset.estimateWind(['w_y','w_x','w_z'])
        
        # Plot
        dataset.plot()
        
        # Save fix data
        # time, y_lift, y_lift, altitude
        dataset.saveToDB()

        # Calculate minmax altitude for filtering
        dataset.calculateMinmaxAltitude()
                
        # Define lift categories

        # Divide to categories
        
        # Calculate lift locations

        # Save flight data 

        # Calculate weights / resample

        # Print status
        count += 1
        print("\r",count," flight processed", end='')

    # Close database
    conn.close()
    print("\nCompleted")

def EstimateAcceleration(dataframe, key, new_keys, c = 0): 
    y_spl = UnivariateSpline(dataframe['Time'],dataframe[key])    
    y_spl_1d = y_spl.derivative(n=1)
    y_spl_2d = y_spl.derivative(n=2)
    
    dataframe[new_keys[0]] = y_spl_1d(dataframe['Time'])
    dataframe[new_keys[1]] = y_spl_2d(dataframe['Time']) + c

def CalculateMinmaxAltitude(dataframe):
    dataframe['max_from_start'] = dataframe['z'].cummax()
    dataframe['max_from_end'] = dataframe['z'].iloc[::-1].cummax().iloc[::-1]
    dataframe['Minmax_Altitude']=dataframe[['max_from_end', 'max_from_start']].min(axis=1)
    dataframe.drop(['max_from_start', 'max_from_end'], axis=1, inplace=True)

def EstimateWind(dataframe, keys):
    n = len(dataframe.index) 
    
    # Set glider performance (TODO estimate) 
    k0 = 1.2 / 10000
    k1 = 1.3 * 100 / (9.81 ** 2)

    # Get vector
    a = dataframe[['a_x','a_y','a_z']].to_numpy()
    v = dataframe[['v_x','v_y','v_z']].to_numpy()

    # Iterate trough the flight
    w = zeros((n, 3))
    dw = zeros(n)
    q = array([-1, 0, 3, 0, -3, 0, 0])
    
    for i in range(1,n):
        # Calculate TAS (pre)
        TAS = v[i,:] - w[i-1,:]
        norm_TAS = sqrt(dot(TAS, TAS))
        norm_a = sqrt(dot(a[i,:],a[i,:]))

        if norm_TAS > 15:
            # Solve change of wind  
            v_a_pre = dot(a[i,:], TAS) / norm_a
            v_T = TAS - dot(a[i,:], TAS) / dot(a[i,:], a[i,:]) * a[i,:]
            norm_v_T = sqrt(dot(v_T, v_T))
            q[5] = (norm_v_T ** 2) / (k1 * norm_a)
            q[6] = k0 * (norm_v_T ** 4) / (k1 * (norm_a ** 2)) + 1 
            r = roots(q)
            cos_theta = r[isreal(r)].min().real
            if cos_theta < -1 or cos_theta > 1:
                print('bad wind update')
            
            dw[i] =  v_a_pre - norm_TAS * cos_theta
        else:
            dw[i] =  0
        # Update wind
        w[i,:] = w[i-1, :] +  0.1 * dw[i] / norm_a * a[i,:] 

    # Save values
    dataframe['w_x'] = w[:,0]
    dataframe['w_y'] = w[:,1]
    dataframe['w_z'] = w[:,2]

    # Print update
    print('Average wind update: {0:.2f}'.format(sum(abs(dw))/n))

def CalculateWindChange(k, v, a):
    return sum(dw), w

if __name__ == '__main__':
    databaseName = "LiftData_20201017_171445"
    RecalculateAll = True 
    main(databaseName, RecalculateAll)