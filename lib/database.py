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
            #self.cursor.execute('CREATE TABLE Estimates(FlightID TEXT, Timestamp INT, Latitude FLOAT, Longitude FLOAT, Altitude FLOAT, XWind FLOAT, YWind FLOAT, ZWind FLOAT, GS_x FLOAT GS_y FLOAT, GS_z FLOAT, theta FLOAT, n FLOAT, PRIMARY KEY(FlightID, Timestamp))')
            #self.cursor.execute('CREATE TABLE FilterStates(FlightID TEXT, Timestamp INT, Variable TEXT, Value FLOAT, PRIMARY KEY(FlightID, Timestamp))')

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

    def command(self, queryString):
        
        self.cursor.execute(queryString)
        self.file.commit()

    def clearEstimateTable(self):
        # Drop tables
        self.cursor.execute("DROP TABLE Estimates")
        #self.cursor.execute("DROP TABLE FilterStates")

        self.file.commit

        #Create tables
        self.cursor.execute('CREATE TABLE Estimates(FlightID TEXT, Timestamp INT, Latitude FLOAT, Longitude FLOAT, Altitude FLOAT, XWind FLOAT, YWind FLOAT, ZWind FLOAT, GS_x FLOAT, GS_y FLOAT, GS_z FLOAT, theta FLOAT, n FLOAT, PRIMARY KEY(FlightID, Timestamp))')
        #self.cursor.execute('CREATE TABLE FilterStates(FlightID TEXT, Timestamp INT, Variable TEXT, Value FLOAT, PRIMARY KEY(FlightID, Timestamp))')
        self.file.commit()