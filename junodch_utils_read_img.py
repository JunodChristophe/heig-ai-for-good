import numpy as np
#import gdal

import rasterio
from rasterio.merge import merge as rasterMerge
from rasterio.mask import mask as rasterMask
from shapely.geometry import Polygon



def getMosaicFromFiles(filesRaster,meta):
  mosaic, output = rasterMerge(filesRaster)
  meta.update({
      "driver": "GTiff",
      "height": mosaic.shape[1],
      "width": mosaic.shape[2],
      "transform": output,
  })
  return mosaic, meta

def getCoordForPixel(mask, transform):
  dataX, dataY = np.where(mask)
  data = np.c_[dataX, dataY]
  sampleData = []
  for x, y in data:
    p = []
    for offset in ['ul','ur','lr','ll']:
      p.append(rasterio.transform.xy(transform, x, y, offset=offset))
    sampleData.append(p)
  return sampleData

def getImgFromCoord(raster, areas, crop=True):
  pol = []
  for a in areas:
    pol.append(Polygon(a))
  tile, transform = rasterMask(raster, pol, crop=crop)
  return tile, transform

def getTrainingAndTestPerimeter(path, threashold):
  with rasterio.open(path) as file:
    night = file.read(1)
    threashold = 200
    train = getCoordForPixel(night > threashold, file.transform)
    test = getCoordForPixel((0 < night) & (night <= threashold), file.transform)
  return train, test
