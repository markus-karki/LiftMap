from time import gmtime, strftime
import sqlite3
import os 
from pyproj import Proj

# IGC data to database

class Database:
    def __init__(self, databaseName):
        if databaseName == "":
            dbName = strftime("LiftData_%Y%m%d_%H%M%S", gmtime())
        else:
            dbName = databaseName
        
        self.file = sqlite3.connect(dbName)
        
        # Create tables, if new database
        if databaseName == "":
            self.file.cursor.execute('''CREATE TABLE Flights(ID TEXT, Date INT, Type TEXT , PRIMARY KEY(ID))''')
            self.file.cursor.execute('''
            CREATE TABLE Fixes(FlightID TEXT, Timestamp INT, Latitude INT, Longitude INT, Altitude INT, AltitudeFilter INT,PRIMARY KEY(FlightID, Timestamp))''')
            self.file.cursor.execute('''
            CREATE TABLE AirFlowEstimates(Timestamp INT, Latitude INT, Longitude INT, Altitude INT, AltitudeFilter INT,PRIMARY KEY(flight_id, km_id))''')
            self.file.cursor.execute('''
            CREATE TABLE FixEstimates(FlightID TEXT, Timestamp INT, Lat INT, Lon INT, Alt INT, V_lat INT, V_lon INT, V_alt INT, Theta INT, Lift INT, PRIMARY KEY(FlightID, Timestamp))''')
            self.file.commit()

def main(folder, databaseName):
    #Create or open database
    database = Database(databaseName)

    #Open IGC files and save fixes to db
    fileList=os.listdir(folder)
    projection = Proj(proj='utm',zone=35,ellps='WGS84')

    #Go throught files
    for igc_file in filelist:
        #Check if in database
        #TODO

        # Read file
        if igc_file.endswith('.igc'):
            # Read igc and save flight data to database
            f = open(folder+"/"+igc_file, 'r')
            for line in f:
                # Read date
                if  line[0:5]=="HFDTE":
                    flight_date=line[5:7]+"/"+line[7:9]+"/20"+line[9:11]

                # Read type
                if  line[0:5]=="HFGTYGLIDERTYPE":
                    flight_date=line[5:7]+"/"+line[7:9]+"/20"+line[9:11]
                #Check if line is a gps-fix
                if line[0]=="B":
                    # Extract time
                    fix_time=int(line[1:3])*3600+int(line[3:5])*60+int(line[5:7])
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

                    # Save to database
                    cursor.execute('''INSERT INTO fix_table(flight_id, fix_id, date_, time_,  lat, lon,  altitude) VALUES(?,?,?,?,?,?,?)''', (igc_file,fix_id,flight_date,fix_time,int(utm_lat),int(utm_lon),fix_altitude))
                        
            print(igc_file)			
            # Commit to database
            database.file.commit()
    
    database.file.close()

# Parse arguments
if __name__ == '__main__':
    igc_folder = "/Users/Markus/OneDrive/IGC_analysis/EFRYarkisto"
    databaseName = ""

    main(igc_folder,database_name)
