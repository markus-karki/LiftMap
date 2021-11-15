# -*- coding: utf-8 -*-
__author__ = 'Markus Kärki'
__version__ = "0.0.0"

'''
A script for creating visualization on lift data in a sqlite database. Needs database name as input

Updates
    -

Todo
    - plot coastline
    - plot turnpoints
    - plot data
    - plot 

Known issues
    -
'''
from sqlite3 import connect
from pandas import DataFrame, read_csv, read_sql_query

import json
import plotly.express as px
import plotly.graph_objects as go

from osgeo import osr

import rasterio

def soil_data_test():
    #dataset = rasterio.open('./data/wrbfu_directory/wrbfu/hdr.adf')

    dataset = rasterio.open('./data/landcover/DATA/U2018_CLC2018_V2020_20u1.tif')



def main(databaseName):
    # Open database
    conn = connect(databaseName + '.db')
    cursor = conn.cursor()


    # Open turnpoint data

    # Open landcover data
    #https://land.copernicus.eu/pan-european/corine-land-cover/clc2018
   # https://gist.github.com/Turbo87/72d73a548dbf953e1f5e
   #response = requests.get("https://skylines.aero/api/flights/date/2019-05-19")
    # Open soil data
    #https://digital-geography.com/free-global-soil-grids-1km-resolution/
    #https://esdac.jrc.ec.europa.eu/ApplicationAndServices
    
    # Get datapoints    
    #cursor.execute('SELECT * FROM Estimates', ())
    #df = DataFrame(cursor.fetchall())#, columns=['x_lift', 'y_lift', 'isLift'])
    df = read_sql_query('SELECT * FROM Estimates', conn)
    param = json.load(open('.parameters/projection.json'))   
    
    source = osr.SpatialReference()
    source.ImportFromEPSG(3067)

    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)

    p = osr.CoordinateTransformation(source, target)
    
    lon, lat = p.transformPoint(df['x_lift'].tolist(), df['y_lift'].tolist())
    df['Latitude']=lat 
    df['Longitude'] = lon

    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color='isLift', hover_data=['Time','z','w_x','w_y','w_z'])
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.show()

def map_test():

    df_turnpoints = read_csv('finland8.cup', encoding='latin1' )
    
    df_turnpoints['Latitude'] = (df_turnpoints['Latitude'].str[0:2].astype(float) 
        + df_turnpoints['Latitude'].str[2:4].astype(float)/100/.6  
        + df_turnpoints['Latitude'].str[5:8].astype(float)/100000/.6)
    
    df_turnpoints['Longitude'] = (df_turnpoints['Longitude'].str[0:3].astype(float) 
        + df_turnpoints['Longitude'].str[3:5].astype(float)/100/.6 
        + df_turnpoints['Longitude'].str[6:9].astype(float)/100000/.6)
    
    #countries = json.load(open("./countries/CNTR_BN_60M_2020_3035_COASTL.geojson"))
    
    fig = px.scatter_mapbox(df_turnpoints, lat="Latitude", lon="Longitude", hover_name="Title", hover_data=["Description"])
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.show()

if __name__ == '__main__':
    databaseName = "database"
    #main(databaseName)
    #map_test()
    soil_data_test()