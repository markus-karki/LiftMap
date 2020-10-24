import sqlite3
import json 
from time import mktime
from datetime import date
from os import listdir
from os.path import exists
from pyproj import Proj

class Folder:
    def __init__(self, folderPath):
        self.path = folderPath
        self.files = listdir(folderPath)
        self.nFiles = len(self.files)
        self.status = {'added': 0, 'found': 0, 'rejected':0}
        print(self.nFiles, " files found")
    
    def updateStatus(self, group):
        self.status[group] = self.status[group] + 1
        percentage = int(sum(self.status.values()) * 100 / self.nFiles)
        print(percentage,str(chr(37))," completed", end='\r')
    
    def printStatus(self):
        print("Completed: ", self.status['added'], "files added, ", self.status['found']," files already in database, ", self.status['rejected']," non igc-files skipped")

class File:
    def __init__(self, folder, igc_file):
        self.file = open(folder.path+"/"+igc_file, 'r', encoding='latin-1')
        self.flightID = igc_file.rsplit('.',1)[0]
        self.date = 0
        self.gliderType = "Unknown" 
        self.fix = list(tuple())
        self.time = 0
    
    def readData(self, projection):
        for line in self.file:
            #Check if line is a gps-fix
            if line[0]=="B":
                # Extract time
                fix_time = int(line[1:3])*3600+int(line[3:5])*60+int(line[5:7])
                timestamp = fix_time + self.date
                
                if timestamp != self.time:
                    self.time = timestamp
                    # Extract lat 
                    wgs_lat=float(line[7:9])+float(line[9:14])/60000
                    
                    # Extract lon 
                    wgs_lon=float(line[15:18])+float(line[18:23])/60000
                    
                    #Convert coordinates
                    utm_lon, utm_lat = projection.convert(wgs_lon,wgs_lat)

                    # Extract altitude
                    if line[25:30]=="000000":
                        fix_altitude=int(line[30:35])
                    else:
                        fix_altitude=int(line[25:30])

                    self.fix.append((self.flightID, timestamp, int(utm_lat), int(utm_lon), fix_altitude))   
            
            # Read date
            elif line[0:5]=="HFDTE":
                # date = line[5:7]+"/"+line[7:9]+"/20"+line[9:11]
                y = int(line[9:11])+2000
                m = int(line[7:9])
                d = int(line[5:7])
                dt = date(y,m,d)
                self.time = int(mktime(dt.timetuple()))
                self.date = self.time

            # Read type
            elif line[0:16]=="HFGTYGLIDERTYPE":
                self.gliderType = line[16:]   

class Projection:
    def __init__(self, param):
        self.f = Proj(param)

    def convert(self, longitude, latitude):
        return self.f(longitude, latitude)

class Database:
    def __init__(self, databaseName):
        databaseExists = exists('./'+ databaseName + '.db')
        
        self.file = sqlite3.connect(databaseName+ '.db')
        self.cursor=self.file.cursor()

        # Create tables, if new database
        if not(databaseExists):
            print('Database created')
            self.cursor.execute('CREATE TABLE Flights(FlightID TEXT, Date INT, Type TEXT , PRIMARY KEY(FlightID))')
            self.cursor.execute('CREATE TABLE Fixes(FlightID TEXT, Timestamp INT, Latitude INT, Longitude INT, Altitude INT, PRIMARY KEY(FlightID, Timestamp))')
            self.file.commit()
    
    def insertFile(self, file):
        self.cursor.execute('INSERT INTO Flights(FlightID, Date, Type) VALUES(?,?,?)',(file.flightID,file.date,file.gliderType))
        self.file.commit()

    def insertFixes(self, file):
        self.cursor.executemany('INSERT or IGNORE INTO Fixes VALUES(?,?,?,?,?)', file.fix)
        self.file.commit()
    
    def fileExists(self, file):
        self.cursor.execute('SELECT EXISTS(SELECT FlightID FROM Flights WHERE FlightID=?)', (file.flightID,))
        data = self.cursor.fetchall()
        return data[0][0]

# IGC data to database
def main(folderPath, databaseName):
    # Create or open database
    database = Database(databaseName)

    # Open IGC files
    folder = Folder(folderPath)
    
    # Create projection
    param = json.load(open('projection.json'))
    projection = Projection(param)
    
    # Go through files
    for igcFile in folder.files:
        # Check if .igc-file
        isIGC = igcFile.endswith('.igc')
        if not(isIGC):
            folder.updateStatus('rejected')
        else:
            # Check if in database, add if not
            file = File(folder, igcFile)
            existsInDatabase = database.fileExists(file)
            if existsInDatabase:
                folder.updateStatus('found')
            else:
                # Read data and save to database
                file.readData(projection)
                database.insertFile(file)
                database.insertFixes(file)
                folder.updateStatus('added')

    database.file.close()
    folder.printStatus()

# Parse arguments
if __name__ == '__main__':
    igc_folder = "/Users/Markus/OneDrive/IGC_analysis/TestiArkisto"
    databaseName = "test3"

    main(igc_folder,databaseName)