# Open lift database

import rasterio
import rasterio.crs
from rasterio.transform import Affine

import json
from sklearn.linear_model import LogisticRegression

from sqlite3 import connect


from pandas import DataFrame, read_sql_query
from numpy import zeros, unique, save, load, sort, meshgrid, linspace, reshape, flipud, ones, minimum, maximum

from osgeo import osr

#import plotly.express as px

import pickle
        
class Geodata:
    def __init__(self, geodatasets):
        source = osr.SpatialReference()
        source.ImportFromEPSG(32635)
        target = osr.SpatialReference()
        target.ImportFromEPSG(3035)
        self.utm_transformer =  osr.CoordinateTransformation(source, target)
        source = osr.SpatialReference()
        source.ImportFromEPSG(3857)
        self.webMerc_transformer =  osr.CoordinateTransformation(source, target)
        self.dataset = rasterio.open(geodatasets)
        self.data = self.dataset.read(1)
    
    def getFeatureVector(self, lon, lat, coord_type='utm'):
        n = lon.shape[0]
        X = zeros((n, 256))
        a = 5
        for i in range(n):
            if coord_type == 'webMercator':
                [y, x, z] = self.webMerc_transformer.TransformPoint(lon[i], lat[i])
            else:
                [y, x, z] = self.utm_transformer.TransformPoint(lon[i], lat[i])
            [x_ind, y_ind] = self.dataset.index(x,y)
            [values, counts] = unique(self.data[(x_ind-a):(x_ind+a), (y_ind-a):(y_ind+a)], return_counts=True)
            X[i, values] = counts / ((2*a)**2)
        return X

class Dataset:
    def __init__(self, databaseName):
        conn = connect(databaseName + '.db')
        self.data = read_sql_query('SELECT * FROM Estimates', conn)

    def getTrainingData(self, geodata):
        y = self.data['isLift'].round().to_numpy()
        X = geodata.getFeatureVector(self.data['x_lift'].to_numpy(), self.data['y_lift'].to_numpy())
        return X, y

def colorMap(Z):
    midpoint = 0.3 * 255
    out = zeros((Z.shape[0],Z.shape[1], 3))
    one = ones((Z.shape[0],Z.shape[1]))*255
    out[:,:,0] = minimum(Z/midpoint*255, one)
    out[:,:,1] = maximum(255 - abs(Z-midpoint), one*0)
    out[:,:,2] = minimum(-255/(255-midpoint)*Z+(255*255)/(255-midpoint), one)
    return out.astype('uint8')

def trainModel(database_name, geodatasets):
    # create geodata
    geodata = Geodata(geodatasets)

    # open database
    dataset = Dataset(database_name)
    [X, y] = dataset.getTrainingData(geodata)

    # Train model
    clf = LogisticRegression(random_state=0).fit(X, y)

    # Print score
    #p=clf.predict_proba(X)
    #p=sort(p[:,1])
    #fig = px.line(p)
    #fig.show()

    # Save parameters
    return clf

def predictLift(geodatasets, clf):
    # create geodata
    geodata = Geodata(geodatasets)    

    # Calculate georaster
    x_min = 2500000
    x_n = 300
    x_max = 2800000

    y_min = 8400000
    y_n = 300
    y_max = 8700000
    

    [xx, yy] = meshgrid(linspace(x_min, x_max, x_n),linspace(y_min, y_max, y_n))

    X = geodata.getFeatureVector(reshape(xx, (-1,)), reshape(yy,(-1,)), 'webMercator')

    #Open model
    zz = clf.predict_proba(X)

    Z = flipud(reshape(zz[:,1]*255, (y_n, x_n))).astype('uint8')
    Z = colorMap(Z)
    # Save results
    res = (x_max - x_min) / x_n
    transform = Affine.translation(x_min - res / 2, y_max + res / 2) * Affine.scale(res, - res)

    new_dataset = rasterio.open(
        './visualizations/new2.tif',
        'w',
        driver='GTiff',
        height=Z.shape[0],
        width=Z.shape[1],
        count=3,
        dtype=Z.dtype,
        crs=rasterio.crs.CRS({"init": "epsg:3857"}),
        transform=transform
        )
    new_dataset.write(Z[:,:,0], 1)
    new_dataset.write(Z[:,:,1], 2)
    new_dataset.write(Z[:,:,2], 3)
    new_dataset.close()

if __name__ == '__main__':
    datasets = './data/landcover/DATA/U2018_CLC2018_V2020_20u1.tif'
    database_name = 'database'
    clf = trainModel(database_name, datasets)
    predictLift(datasets, clf)