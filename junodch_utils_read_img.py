
import math
import matplotlib.pyplot as matPlt
import matplotlib.gridspec as gridspec
import numpy as np

import rasterio
from rasterio import plot as rastPlt
from rasterio.merge import merge as rasterMerge
from rasterio.mask import mask as rasterMask
from rasterio.plot import reshape_as_raster, reshape_as_image
from shapely.geometry import Polygon, box
from sklearn.metrics import confusion_matrix

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

# Get the 4 corners coordinates of the image
def getImgBorder(pathImg):
  # Filter the night tile not covering the day area.
  with rasterio.open(pathImg) as f:
    sBox = box(*f.bounds).exterior.coords
  aoi = []    # Area of interest
  for v in sBox:
    aoi.append((v[0], v[1]))
  aoi.pop() # Remove unnecessary duplicate (The first and the last are the same point.)
  return aoi

# Get the 4 corners coordinate for the given pixels
def getCoordsForPixels(data, transform):
  corners = np.array(['ul','ur','lr','ll'])
  sampleData = np.empty([data.size, corners.size, 2])
  for (x, y), _ in np.ndenumerate(data):
    p = np.empty([4, 2])
    for i, offset in enumerate(corners):
      p[i, :] = rasterio.transform.xy(transform, x, y, offset=offset)
    sampleData[x*data.shape[1] + y, :] = p
  return sampleData

# From the path to a raster, get the perimeter of each valid pixel within the area of interest
# Also return the radiance of each pixels. radianceAmplifier reduce the gaps between values.
def getTilesCoordsPerimeter(path, area=None, radianceAmplifier=2):
  with rasterio.open(path) as file:
    if area is not None: # Filter area of interest
      gridValues, transform  = getImgFromCoord(file, [area], True)
      gridValues = gridValues[0] # Only take one set of color. (Assuming the image is black and white.)
    else: # Read all
      gridValues = file.read(1)
      transform = file.transform
  data = getCoordsForPixels(gridValues, transform)
  
  # Radiance normalization
  dataRadiance = gridValues.flatten()
  dataRadiance = np.log((dataRadiance.astype('float')*radianceAmplifier+1)) # Attempt to reduce the difference between low values and high values.
  dataRadiance = dataRadiance / dataRadiance.max()
  return data, dataRadiance

# From raster get the area of interest
def getImgFromCoord(raster, areas, crop=True):
  pol = []
  for a in areas:
    pol.append(Polygon(a))
  tile, transform = rasterMask(raster, pol, crop=crop)
  return tile, transform

def coordsToImgsFormated(fileOpen, areasCoords, res=64):
  tiles = np.empty([len(areasCoords), res, res, 3])
  meta = []
  for i, a in enumerate(areasCoords):
    tile, transform = rasterMask(fileOpen, [Polygon(a)], crop=True)
    if tile.shape[1] < res or tile.shape[2] < res:
      pad1 = res - tile.shape[1] if tile.shape[1] < res else 0
      pad2 = res - tile.shape[2] if tile.shape[2] < res else 0
      tile = np.pad(tile, ((0,0),(0, pad1),(0, pad2)))
    tiles[i, :, :, :] = reshape_as_image(tile[:3,:res,:res]).astype("float32") / 255.0
    meta.append(transform)
  return tiles, meta

# Get each individual image from multiple area of interest
def getEachImgFromCoord(raster, areas, crop=True):
  tiles = []
  meta = []
  for a in areas:
    tile, transform = rasterMask(raster, [Polygon(a)], crop=crop)
    tiles.append(tile)
    meta.append(transform)
  return tiles, meta

# set shape to res, move the channel dimension to the end of the shape and convert values between [0,1] if toFloat is True
def formatData(data, res=64, toFloat=False):

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

def displayTiles(tile, meta, ax=None):
  if ax == None:
    fig, ax = matPlt.subplots(1,1)
  rastPlt.show(tile[0],transform=meta[0],ax=ax)
  xMin = ax.get_xlim()[0]
  xMax = ax.get_xlim()[1]
  yMin = ax.get_ylim()[0]
  yMax = ax.get_ylim()[1]

  for i in range(1,len(tile)):
    rastPlt.show(tile[i],transform=meta[i],ax=ax)
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

# Display the original images and the result output of the autoencoder.
# showDetail 0 : Show only output
# showDetail 1 : Show encoded data and output
# showDetail 2 : Show all intermediate data (If showable on a 2d plane)
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
  encoder = keras.Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer('displayable_encoder').output)
  encoded_imgs = encoder.predict(dataInput)
  displayImgCollection(encoded_imgs)

# Display all layer from autoencoder
def displayDetailAutoencoder(autoencoder, dataInput):
  layers = autoencoder.layers[0:len(autoencoder.layers)-1]
  for l in layers:
    # check if this is a type of layer that can be display on a 2d plane
    if 'Conv2D' in l.__class__.__name__:
      intermediateLayers = keras.Model(inputs=autoencoder.inputs, outputs=l.output)
      encoded_imgs = intermediateLayers.predict(dataInput)

      displayImgCollection(encoded_imgs)

# Display list of images
def displayImgs(dataImg, titles=None):
  # title : list of title to write at the top of each plot.
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

# Display images that are grouped on a grid.
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
def scanSatellite(pathSat, coords, calcScore, batch=256, res=64):
  iSave = 0
  iNext = 0

  size = len(coords)
  result = np.asarray([0.0]*size)

  with rasterio.open(pathSat) as f:
    while iNext < size:
      iNext = iSave+batch
      if iNext > size:
        iNext = size
      
      data, _ = coordsToImgsFormated(f, coords[iSave:iNext], res=res)
      score = calcScore(data)
      result[iSave:iNext] = score
      iSave = iNext
  
  return result

# key1 : is detected
# key2 : is in settlement data
# key3 : has light
defaultColorMap = {
  (0,0,0): (0,0,0),
  (0,0,1): (0,0,0),
  (0,1,0): (255,0,255),
  (0,1,1): (255,0,0),
  (1,0,0): (0,0,255),
  (1,0,1): (0,255,0),
  (1,1,0): (0,255,0),
  (1,1,1): (0,255,0),
}

# Create a new image colored from the scoreTiles and validTiles.
def mapResultOnImg(pathTemplate, coordsTiles, scoreTiles, validTiles, lightTiles, colorMap=defaultColorMap):
  def getCenter(corners):
    lon = corners[0][0] - (corners[0][0] - corners[1][0])/2
    lat = corners[1][1] - (corners[1][1] - corners[2][1])/2
    return lon, lat
  
  with rasterio.open(pathTemplate) as s:
    data = s.read()
    meta = s.transform
    data[:] = 0   # blank content
    
    for i, c in enumerate(coordsTiles):
      cx,cy = getCenter(c)
      px,py = s.index(cx,cy)
      rgb = colorMap[(scoreTiles[i], validTiles[i], lightTiles[i])]
      
      data[0][px][py] = rgb[0]
      data[1][px][py] = rgb[1]
      data[2][px][py] = rgb[2]
      data[3][px][py] = 255
  
  return data, meta

def processConfusionMatrix(detection, validation, light):
  confusionMatrix = confusion_matrix(light, detection)
  tp = confusionMatrix[1][1]
  fp = confusionMatrix[0][1]
  fn = confusionMatrix[1][0]
  print('Total light data:',tp+fn,'Detected:',tp,'Missed:',fn)
  print('Population with light detected:',round(tp / (tp + fn) * 100,2),"%")
  print('')
  
  print('Process confustion matrix...')
  print('total data:',len(detection))
  confusionMatrix = confusion_matrix(validation, detection)
  print(confusionMatrix)
  tp = confusionMatrix[1][1]
  fp = confusionMatrix[0][1]
  fn = confusionMatrix[1][0]
  print('f-score:',round(tp / (tp + (fp + fn)/2) * 100, 2),"%")

def rasterToImg(image, pathTemplate):
  img = np.copy(image)
  img = img[0:3,:,:]
  img = img.transpose([1, 2, 0])
  with rasterio.open(pathTemplate) as f:
    profile = f.profile
  img = reshape_as_raster(img)
  profile.update(count=3)
  return img, profile