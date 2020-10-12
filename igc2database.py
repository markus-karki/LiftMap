import time
import datetime
import sqlite3
import os 
from pyproj import Proj
from lib.database import Database

# IGC data to database

def main(folder, databaseName):
    #Create or open database
    database = Database(databaseName)

    #Open IGC files and save fixes to db
    fileList=os.listdir(folder)
    print(len(fileList)," files found")
    projection = Proj(proj='utm',zone=35,ellps='WGS84')
    status = {'added': 0, 'found': 0, 'rejected':0, 'done':0}
    
    # Go through files
    for igc_file in fileList:
        flightID = igc_file.rsplit('.',1)[0]
        # Check if in database
        if database.getData('SELECT EXISTS(SELECT FlightID FROM Flights WHERE FlightID=?)',(flightID,))[0][0]:
            status['found']=status['found'] + 1
            status['done']=status['done'] + 1

        # Check if .igc-file
        elif not(igc_file.endswith('.igc')):
            status['rejected']=status['rejected'] + 1
            status['done']=status['done'] + 1

        # Add file
        else:
            status['added']=status['added'] + 1
            status['done']=status['done'] + 1

        # Read igc and save flight data to database
            f = open(folder+"/"+igc_file, 'r', encoding='latin-1')
            date = "Unknown"
            gliderType = "Unknown" 
            currentData = list(tuple())
            altitude_filter = 0 
            previous_time = 0

            for line in f:
                # Read date
                if  line[0:5]=="HFDTE":
                    # date = line[5:7]+"/"+line[7:9]+"/20"+line[9:11]
                    dt = datetime.date(int(line[9:11])+2000,int(line[7:9]),int(line[5:7]))
                    date = int(time.mktime(dt.timetuple()))
                # Read type
                if  line[0:5]=="HFGTYGLIDERTYPE":
                    gliderType = line[16:]
                #Check if line is a gps-fix
                if line[0]=="B":
                    # Extract time
                    fix_time = int(line[1:3])*3600+int(line[3:5])*60+int(line[5:7])
                    timestamp = fix_time + date
                    
                    if timestamp != previous_time:
                        previous_time = timestamp
                        # Extract lat INTEGER 
                        wgs_lat=float(line[7:9])+float(line[9:14])/60000
                        # Extract lon INTEGER
                        wgs_lon=float(line[15:18])+float(line[18:23])/60000
                        #Convert coordinates
                        utm_lon, utm_lat=projection(wgs_lon,wgs_lat)

                        # Extract altitude INTEGER: read+convert
                        if line[25:30]=="000000":
                            fix_altitude=int(line[30:35])
                        else:
                            fix_altitude=int(line[25:30])

                        currentData.append((flightID, timestamp, int(utm_lat), int(utm_lon), fix_altitude))
                
            # Save to database
            database.insertData('INSERT INTO Flights(FlightID, Date, Type) VALUES(?,?,?)',(flightID,date,gliderType))
            database.insertManyData('INSERT or IGNORE INTO Fixes VALUES(?,?,?,?,?)', currentData)
        
        print("\r",status['done']," files processed", end='')

    print("\n",status['added'], "files added, ", status['found']," files already in database, ", status['rejected']," files rejected")

    database.file.close()

# Parse arguments
if __name__ == '__main__':
    igc_folder = "/Users/Markus/OneDrive/IGC_analysis/TestiArkisto"
    databaseName = ""

    main(igc_folder,databaseName)