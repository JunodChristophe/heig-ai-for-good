
import matplotlib.pyplot as plt
import numpy as np
#import gdal

import rasterio
from rasterio import plot as rastPlt
from rasterio.merge import merge as rasterMerge
from rasterio.mask import mask as rasterMask
from shapely.geometry import Polygon
import tensorflow as tf

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

def displayTiles(img, meta, ax=None):
  if ax == None:
    fig, ax = plt.subplots(1,1)
  rastPlt.show(img[0],transform=meta[0],ax=ax)
  xMin = ax.get_xlim()[0]
  xMax = ax.get_xlim()[1]
  yMin = ax.get_ylim()[0]
  yMax = ax.get_ylim()[1]

  for i in range(1,len(img)):
    rastPlt.show(img[i],transform=meta[i],ax=ax)
    newXMin = ax.get_xlim()[0]
    newXMax = ax.get_xlim()[1]
    newYMin = ax.get_ylim()[0]
    newYMax = ax.get_ylim()[1]
    
    xMin = newXMin if newXMin < xMin else xMin
    xMax = newXMax if newXMax > xMax else xMax
    yMin = newYMin if newYMin < yMin else yMin
    yMax = newYMax if newYMax > yMax else yMax

  ax.set_xlim((xMin, xMax))
  ax.set_ylim((yMin, yMax))

def formatDataForAutoencoder(data, res=32):
  # fix inconcistant shapes
  for i, d in enumerate(data):
    if d.shape[1] < res or d.shape[2] < res:
      pad1 = res - d.shape[1]
      pad2 = res - d.shape[2]
      data[i] = np.lib.pad(d, ((0,0),(0,pad1 if pad1 > 0 else 0),(0,pad2 if pad2 > 0 else 0)), 'constant', constant_values=(0))
    if d.shape[1] > res or d.shape[2] > res:
      data[i]=data[i][:,0:res,0:res]

  # to numpy
  data = np.asarray(data)
  
  # Transpose shape order to keras expected order.
  data = tf.transpose(data, [0, 2, 3, 1])
  data = tf.slice(
    data,
    [0, 0, 0, 0],
    [len(data),res,res,4])
  return data

def displayAutoEncoderResults(dataInput, encoder, autoencoder, lossFunction):
  MAX_ON_ROW = 20
  total = dataInput.shape[0]
  nRow = (dataInput.shape[0] // MAX_ON_ROW) + 1
  nCol = MAX_ON_ROW if total > MAX_ON_ROW else total

  # Display original
  plt.figure(figsize=(30,nRow*2))
  for i in range(0, total):
    ax = plt.subplot(nRow, nCol, 1+i)
    plt.imshow(dataInput[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  print("Original data:",dataInput.shape)
  plt.show()

  if encoder != None:
    # Display encoded. The first MAX_ON_ROW only
    encoded_imgs = encoder.predict(dataInput[:nCol])
    plt.figure(figsize=(30,encoded_imgs[0].T.shape[0]))
    for i in range(0, encoded_imgs.shape[0]):
      numRow = encoded_imgs[i].T.shape[0]
      for index, img in enumerate(encoded_imgs[i].T):
        ax = plt.subplot(numRow, nCol, 1+i+nCol*index)
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    print("Encoded data:",encoded_imgs.shape)
    plt.show()
  
  # Display reconstruction
  decoded_imgs = autoencoder.predict(dataInput)
  plt.figure(figsize=(30,nRow*2))
  for i in range(0, decoded_imgs.shape[0]):
    ax = plt.subplot(nRow, nCol, 1+i)
    plt.title(int(np.round(lossFunction(dataInput[i], decoded_imgs[i]).numpy(),0)))
    plt.imshow(decoded_imgs[i].astype('uint8'))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  print("Output data:",decoded_imgs.shape)
  plt.show()
