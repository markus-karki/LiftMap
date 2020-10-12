import time
import sqlite3

class Database:
    def __init__(self, databaseName):
        if databaseName == "":
            dbName = time.strftime("LiftData_%Y%m%d_%H%M%S", time.gmtime())
            print('Database created')
        else:
            dbName = databaseName
            print(databaseName, " opened")
        
        self.file = sqlite3.connect(dbName)
        self.cursor=self.file.cursor()

        # Create tables, if new database
        if databaseName == "":
            self.cursor.execute('CREATE TABLE Flights(FlightID TEXT, Date INT, Type TEXT , PRIMARY KEY(FlightID))')
            self.cursor.execute('CREATE TABLE Fixes(FlightID TEXT, Timestamp INT, Latitude INT, Longitude INT, Altitude INT, PRIMARY KEY(FlightID, Timestamp))')
            self.cursor.execute('CREATE TABLE Estimates(FlightID TEXT, Timestamp INT, Latitude INT, Longitude INT, Altitude INT, XWind INT, YWind INT, ZWind INT, Speed INT, Direction INT, VerticalSpeed INT, Theta INT, Lift INT, PRIMARY KEY(FlightID, Timestamp))')
            self.file.commit()
            
    def getData(self, queryString, dataTuple):
        self.cursor.execute(queryString, dataTuple)
        data = self.cursor.fetchall()
        return data
    
    def insertData(self, queryString, dataTuple):
        self.cursor.execute(queryString, dataTuple)
        self.file.commit()

    def insertManyData(self, queryString, dataTuple):
        self.cursor.executemany(queryString, dataTuple)
        self.file.commit()

    def clearEstimateTables(self):
        # Drop tables
        self.cursor.execute("DROP TABLE Estimates")
        self.file.commit

        #Create tables
        self.cursor.execute('CREATE TABLE Estimates(FlightID TEXT, Timestamp INT, Latitude INT, Longitude INT, Altitude INT, XWind INT, YWind INT, ZWind INT, Speed INT, Direction INT, VerticalSpeed INT, Theta INT, Lift INT, PRIMARY KEY(FlightID, Timestamp))')
        self.file.commit()