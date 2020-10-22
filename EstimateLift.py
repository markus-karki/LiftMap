#from lib.database import Database
#from lib.flightModel_dummy_2 import Model
#from math import sqrt, acos, cos, sin
from numpy import zeros, matmul, array, empty, roots, real, isreal, sqrt, dot, sum
from scipy.interpolate import UnivariateSpline

from pandas import DataFrame 
import matplotlib.pyplot as plt
from sqlite3 import connect
#from time import strftime, gmtime

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

        # Estimate speed and acceleration
        EstimateAcceleration(df, 'Latitude', ['v_y','a_y'])
        EstimateAcceleration(df, 'Longitude', ['v_x','a_x'])
        EstimateAcceleration(df, 'Altitude',  ['v_z','a_z'], 9.81)
        #CalculateTotalLenght(df, ['a_x','a_y','a_z'], 'a')
        #CalculateTotalLenght(df, ['v_x','v_y','v_z'], 'v')
    
        # Estimate wind
        EstimateWind(df, ['w_y','w_x','w_z'])
        CalculateMinmaxAltitude(df)

        # Save to database
        df.plot(x='Time',subplots=1, layout=(5,3), sharex=1, grid=1)
        plt.show()
        df.to_sql('Estimates', conn, if_exists='append', index = False)

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
    
    #plt.plot(dataframe['Time'],y_spl_1d(dataframe['Time']))
    #plt.show()
    #plt.plot(dataframe['Time'],y_spl_2d(dataframe['Time']))
    #plt.show()
    
    dataframe[new_keys[0]] = y_spl_1d(dataframe['Time'])
    dataframe[new_keys[1]] = y_spl_2d(dataframe['Time']) + c

def CalculateMinmaxAltitude(dataframe):
    dataframe['max_from_start'] = dataframe['Altitude'].cummax()
    dataframe['max_from_end'] = dataframe['Altitude'].iloc[::-1].cummax().iloc[::-1]
    dataframe['Minmax_Altitude']=dataframe[['max_from_end', 'max_from_start']].min(axis=1)
    dataframe.drop(['max_from_start', 'max_from_end'], axis=1, inplace=True)

def CalculateTotalLenght(dataframe, keys, key):
    dataframe[key] = 0
    for i in range(len(keys)):
        dataframe[key] = dataframe[key] + (dataframe[keys[i]] ** 2)
    dataframe[key] = dataframe[key] ** 0.5

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

        # Solve change of wind  
        v_a_pre = dot(a[i,:], TAS) / norm_a
        v_T = TAS - dot(a[i,:], TAS) / dot(a[i,:], a[i,:]) * a[i,:]
        norm_v_T = sqrt(dot(v_T, v_T))
        q[5] = (norm_v_T ** 2) / (k1 * norm_a)
        q[6] = k0 * (norm_v_T ** 4) / (k1 * (norm_a ** 2)) + 1 
        r = roots(q)
        cos_theta = r[isreal(r)].min().real
        if 0:#cos_theta < -1 or cos_theta > 1:
            dw[i] = 0
        else:    
            dw[i] =  v_a_pre - norm_TAS * cos_theta
        
        # Update wind
        w[i,:] = w[i-1, :] +  0.1 * dw[i] / norm_a * a[i,:] 

    # Save values
    dataframe['w_x'] = w[:,0]
    dataframe['w_y'] = w[:,1]
    dataframe['w_z'] = w[:,2]

def CalculateWind(k, v, a):
    len_a = 0
    len_v = 0

    # Iterate trough the flight
    w = zeros((n, 3))
    dw = zeros(n)
    for i in range(1,n):
        # Calculate TAS (pre)
        TAS = v[i,:] - w[i-1,:]
        
        # Solve change of wind  
        v_ = dot(a[i,:], TAS) / len_a[i]
        t1 = (len_v[i] ** 2) / (2 * k1 * len_a[i])
        t2 = sqrt(((len_v[i]**4) * (1 + 4 * k1 * k0))/(4 * k1 * (len_a[i]**2)) + 1)
        cos_theta_ = t1 + t2
        cos_theta = t1 - t2
        dw[i] = len_v[i] * cos_theta - v_
        
        # Update wind
        w[i,:] = w[i-1, :] + 0.02 * dw[i] / len_a[i] * a[i,:] 

    return sum(dw), w

if __name__ == '__main__':
    
    databaseName = "LiftData_20201017_171445"
    #parameters = {saveFilterStates = True, recalculateAll = True}
    main(databaseName)