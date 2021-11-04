import json
import requests

class Database():
    def __init__(self, databaseName):
    
    def init_tables(self, queryString, dataTuple):

class Api():

class Flight():
     def __init__(self, databaseName):
        self.date =
        self.type = 
        self.fix = []

class Fix():
    def __init__(self, databaseName):
        self.time =
        self.lat = 
        self.lon =
        self.alt = 

def main():
    # Connect to database


    # Get list of files

    # Go throught files and add fixes to db


response = requests.get("https://skylines.aero/api/flights/date/2021-09-06")
data = response.json()

for i in range(data['count']):
    filename = data['flights'][i]['igcFile']['filename']
    igc_file = requests.get("https://skylines.aero/files/"+filename)
    file_data = igc_file.text.splitlines()


if __name__ == '__main__':
    igc_folder = "./data/flights"
    databaseName = ""

    main(igc_folder,databaseName)