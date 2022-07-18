
import math
import matplotlib.pyplot as matPlt
import matplotlib.gridspec as gridspec
import numpy as np

import rasterio
from rasterio import plot as rastPlt
from rasterio.merge import merge as rasterMerge
from rasterio.mask import mask as rasterMask
from shapely.geometry import Polygon, box

import tensorflow as tf
import keras

def getMosaicFromFiles(filesRaster,meta):
  mosaic, output = rasterMerge(filesRaster)
  meta.update({
      "driver": "GTiff",
      "height": mosaic.shape[1],
      "width": mosaic.shape[2],
      "transform": output,
  })
  return mosaic, meta

def getImgBorder(pathImg):
  # Filter the night tile not covering the day area.
  with rasterio.open(pathImg) as f:
    sBox = box(*f.bounds).exterior.coords
  aoi = []    # Area of interest
  for v in sBox:
    aoi.append((v[0], v[1]))
  # Remove unnecessary duplicate (The first and the last are the same point.)
  aoi.pop()
  return aoi

# Get the 4 corners coordinate for the given pixels
def getCoordsForPixels(mask, transform):
  dataX, dataY = np.where(mask)
  data = np.c_[dataX, dataY]
  sampleData = []
  for x, y in data:
    p = []
    for offset in ['ul','ur','lr','ll']:
      p.append(rasterio.transform.xy(transform, x, y, offset=offset))
    sampleData.append(p)
  return np.asarray(sampleData)

# From the path to a raster, get the perimeter of each valid pixel within the area of interest
def getTilesCoordsPerimeter(path, validThreshold=1, area=None):
  with rasterio.open(path) as file:
    if area is not None: # Filter area of interest
      gridValues, transform  = getImgFromCoord(file, [area], True)
      gridValues = gridValues[0]
    else: # Read all
      gridValues = file.read(1)
      transform = file.transform
  mask = (validThreshold <= gridValues)
  data = getCoordsForPixels(mask, transform)
  return data, gridValues[mask]

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

# From the path to a raster, get the pixel for training and test data within the area of interest
# OBSOLETE
def getTrainingAndTestPerimeter(path, threashold, area=None):
  with rasterio.open(path) as file:
    if area is not None:
      tile, transform  = getImgFromCoord(file, [area], True)
      tile = tile[0]
    else:
      tile = file.read(1)
      transform = file.transform
  train = getCoordsForPixels(tile > threashold, transform)
  test = getCoordsForPixels((0 < tile) & (tile <= threashold), transform)
  return train, test

def displayTiles(img, meta, ax=None):
  if ax == None:
    fig, ax = matPlt.subplots(1,1)
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

# set shape to res, move the channel dimension to the end of the shape and convert values between [0,1] if toFloat is True
def formatData(data, res=32, toFloat=False):

  # fix inconcistant shapes
  for i, d in enumerate(data):
    if d.shape[1] < res or d.shape[2] < res:
      pad1 = res - d.shape[1]
      pad2 = res - d.shape[2]
      data[i] = np.lib.pad(d, ((0,0),(0,pad1 if pad1 > 0 else 0),(0,pad2 if pad2 > 0 else 0)), 'constant', constant_values=(0))
    data[i]=data[i][:3,:res,:res]
  data = np.asarray(data)
  
  if toFloat:
    data = data.astype("float32") / 255.0
  
  # Transpose shape order to keras expected order.
  data = data.transpose([0, 2, 3, 1])
  return data

# OBSOLETE
def formatDataForAutoencoder(data, res=32, toFloat=True):
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
  if toFloat:
    data = data.astype("float32") / 255.0
  
  # Transpose shape order to keras expected order.
  data = tf.transpose(data, [0, 2, 3, 1])
  data = tf.slice(
    data,
    [0, 0, 0, 0],
    [data.shape[0],res,res,3])
  return data

def displayAutoEncoderResults(autoencoder, dataInput, showDetail=0, precision=0):
  
  print("Original data:",dataInput.shape)
  displayImgs(dataInput)

  if showDetail == 1:
    displayImgEncoder(autoencoder, dataInput)
  elif showDetail == 2:
    displayDetailAutoencoder(autoencoder, dataInput)

  decoded_imgs = autoencoder.predict(dataInput)
  print("Output data:",decoded_imgs.shape)

  scores = ['']*decoded_imgs.shape[0]
  for i in range(len(scores)):
    scores[i] = str(np.round(autoencoder.loss(dataInput[i], decoded_imgs[i]),precision))

  displayImgs(decoded_imgs, scores)

def displayImgEncoder(autoencoder, dataInput):
  encoder = keras.Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer('encoder').output)
  encoded_imgs = encoder.predict(dataInput)
  displayImgCollection(encoded_imgs)

# Display all layer from autoencoder
def displayDetailAutoencoder(autoencoder, dataInput):
  layers = autoencoder.layers[0:len(autoencoder.layers)-1]
  for l in layers:
    if 'Conv2D' in l.__class__.__name__:
      intermediateLayers = keras.Model(inputs=autoencoder.inputs, outputs=l.output)
      encoded_imgs = intermediateLayers.predict(dataInput)
      displayImgCollection(encoded_imgs)

# Obsolete
def displayFormatedImgs(dataInput, createImgTitle=None):
  # createImgTitle : Function to create a title for each individual plot from there index.

  MAX_ON_ROW = 20
  total = len(dataInput)
  nRow = (total // MAX_ON_ROW) + 1
  nCol = MAX_ON_ROW if total > MAX_ON_ROW else total
  
  matPlt.figure(figsize=(30,nRow*2))
  for i in range(0, total):
    ax = matPlt.subplot(nRow, nCol, 1+i)
    if createImgTitle != None:
      matPlt.title(createImgTitle(i))
    matPlt.imshow(dataInput[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  matPlt.show()

# Display list of images
def displayImgs(dataImg, titles=None):
  MAX_ON_ROW = 20
  total = len(dataImg)
  nRow = (total // MAX_ON_ROW) + 1
  nCol = MAX_ON_ROW if total > MAX_ON_ROW else total
  
  matPlt.figure(figsize=(30,nRow*2))
  for i in range(0, total):
    ax = matPlt.subplot(nRow, nCol, 1+i)
    if titles is not None:
      matPlt.title(titles[i])
    matPlt.imshow(dataImg[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  matPlt.show()

def displayImgCollection(imgs):
  grid = gridspec.GridSpec(1, imgs.shape[0])
  matPlt.figure(figsize=(30,imgs[0].T.shape[0]/8))
  for i in range(0, imgs.shape[0]):
    nCol = imgs[i].T.shape[0]
    nRow = 1
    while nCol > 8:
      nCol = math.ceil(nCol/2)
      nRow *= 2
    cell = gridspec.GridSpecFromSubplotSpec(int(nRow), int(nCol), subplot_spec=grid[i], wspace=0.1, hspace=0.1)
    for index, img in enumerate(imgs[i].T):
      ax = matPlt.subplot(cell[index])
      matPlt.imshow(img)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  print("Data:",imgs.shape)
  matPlt.show()


# Validation

# Calculate score for each img. The function is passed with calcScore.
def scanSatellite(pathSat, coords, calcScore, batch=256):
  iSave = 0
  iNext = 0

  size = len(coords)
  result = [0]*size

  with rasterio.open(pathSat) as s:
    while iNext < size:
      iNext = iSave+batch
      if iNext > size:
        iNext = size
      data, _ = getEachImgFromCoord(s, coords[iSave:iNext], True)
      score = calcScore(data)
      result[iSave:iNext] = score
      iSave = iNext
  
  return result

def mapResultOnImg(pathTemplate, coordsTiles, scoreTiles, validTiles):
  def getCenter(corners):
    lon = corners[0][0] - (corners[0][0] - corners[1][0])/2
    lat = corners[1][1] - (corners[1][1] - corners[2][1])/2
    return lon, lat

  def setRGB(val, isInPop):
    rgb = (0,0,0)
    if val != 1: # not settlement
      rgb = (255,0,255) if isInPop else(0,0,255)
    else: # Settlement detected
      rgb = (0,255,0) if isInPop else (255,0,0)
    return rgb
  
  with rasterio.open(pathTemplate) as s:
    data = s.read()
    meta = s.transform
    data[:] = 0   # blank content
    
    for i, c in enumerate(coordsTiles):
      cx,cy = getCenter(c)
      px,py = s.index(cx,cy)
      rgb = setRGB(scoreTiles[i], validTiles[i])
      
      data[0][px][py] = rgb[0]
      data[1][px][py] = rgb[1]
      data[2][px][py] = rgb[2]
      data[3][px][py] = 255
  
  return data, meta