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

# Get the 4 corners coordinate for a given pixel with condition
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

# From raster get the area of interest
def getImgFromCoord(raster, areas, crop=True):
  pol = []
  for a in areas:
    pol.append(Polygon(a))
  tile, transform = rasterMask(raster, pol, crop=crop)
  return tile, transform

# Get each individual image from multiple area of interest
def getEachImgFromCoord(raster, areas, crop=True):
  tiles = []
  meta = []
  for a in areas:
    tile, transform = rasterMask(raster, [Polygon(a)], crop=crop)
    tiles.append(tile)
    meta.append(transform)
  return tiles, meta

# From a path of a raster, get the pixel for training and test data within the area of interest
def getTrainingAndTestPerimeter(path, threashold, area=None):
  with rasterio.open(path) as file:
    if area is not None:
      tile, transform  = getImgFromCoord(file, [area], True)
      tile = tile[0]
    else:
      tile = file.read(1)
      transform = file.transform
  train = getCoordForPixel(tile > threashold, transform)
  test = getCoordForPixel((0 < tile) & (tile <= threashold), transform)
  return train, test
