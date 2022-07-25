from ipyleaflet import *

import numpy as np
import contextily

import junodch_satellites as utilSat
from os.path import exists

from ipywidgets import IntProgress
from IPython.display import display
import time

import ee
from geemap import *

class AOI: # Area of Interest (Rectangle)
  def __init__(self):
    self.isEmpty = False
  
  xMin = 0
  xMax = 0
  yMin = 0
  yMax = 0

class SatelliteData:
  def __init__(self, name, src, vis, zoom):
    self.name = name
    self.src = src
    self.vis = vis
    self.zoom = zoom
    
  def getData(self):
    return 'Not defined'
    
  def getDataWithDate(self, date_start, date_end):
    return 'Not defined'
      
  def getDataWithArea(self, aoi, date_start, date_end):
    return 'Not defined'

class Sentinel2(SatelliteData):
  def __init__(self, name='Sentinel-2',zoom=14):
    self.name = name
    self.src = 'COPERNICUS/S2_SR'
    self.vis = {
      "min": 0, 
      "max": 3000, 
      "bands": ['B4', 'B3', 'B2'],
    }
    self.zoom = zoom
      
  def getData(self):
    return (ee.ImageCollection(self.src)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1)))
  
  def getDataWithDate(self, date_start, date_end):
    return (ee.ImageCollection(self.src)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1))
            .filterDate(date_start, date_end))
  
  def getDataWithArea(self, aoi, date_start, date_end):
    return (ee.ImageCollection(self.src)
            .filterDate(date_start, date_end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1))
            .mosaic().clip(aoi).unmask())
    
class Landsat8_old(SatelliteData):
  # only works for dates under 2022
  def __init__(self, name='Landsat-8', zoom=13):
    self.name = name
    self.src = 'LANDSAT/LC08/C01/T1_SR'
    self.vis = {
      'min': 0, 
      'max': 3000, 
      'bands': ['B4', 'B3', 'B2'],
    }
    self.zoom = zoom
      
  def getData(self):
    return (ee.ImageCollection(self.src)
            .filter(ee.Filter.lt('CLOUD_COVER', 20)))
      
  def getDataWithDate(self, date_start, date_end):
    return (ee.ImageCollection(self.src)
            .filterDate(date_start, date_end)
            .filter(ee.Filter.lt('CLOUD_COVER', 20)))
      
  def getDataWithArea(self, aoi, date_start, date_end):
    return (ee.ImageCollection(self.src)
            .filterDate(date_start, date_end)
            .filter(ee.Filter.lt('CLOUD_COVER', 20))
            .mosaic().clip(aoi).unmask())
    
class NightVIIRS(SatelliteData):
  def __init__(self, minRad=1, maxRad=10, name='Night_VIIRS', zoom=9):
    self.name = name
    self.src = 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'
    self.vis = {
      'min': minRad,
      'max': maxRad, 
      'bands': ['avg_rad'], 
      'palette': ['000000','ffffff'],
    }
    self.zoom = zoom
  
  def getData(self):
    return (ee.ImageCollection(self.src))
      
  def getDataWithDate(self, date_start, date_end):
    return (ee.ImageCollection(self.src)
            .filterDate(date_start, date_end))
      
  def getDataWithArea(self, aoi, date_start, date_end):
    return (ee.ImageCollection(self.src)
            .filterDate(date_start, date_end)
            .mosaic().clip(aoi))

class popGHSL(SatelliteData):
  def __init__(self, name='Population_GHSL', zoom=11):
    self.name = name
    self.src = 'JRC/GHSL/P2016/POP_GPW_GLOBE_V1'
    self.vis = {
      "min": 0.0,
      "max": 20.0,
      "palette": ['000000', 'ffffff'],
    }
    self.zoom = zoom
    
  def getData(self):
    return (ee.ImageCollection(self.src)
            .filterDate('2014-01-01', '2016-01-01')
            ).select('population_count')
  
  def getDataWithDate(self, date_start=None, date_end=None):
    return (ee.ImageCollection(self.src)
            .filterDate('2014-01-01', '2016-01-01')
            ).select('population_count')
      
  def getDataWithArea(self, aoi, date_start, date_end):
    return (ee.ImageCollection(self.src)
            .filterDate('2014-01-01', '2016-01-01')
            .mosaic().clip(aoi)
            ).select('population_count')

def extractCoordFromFeatureCollection(features, precision):
  aoi = AOI()
  aoi.isEmpty = True
  if features.size().getInfo() != 0:
    coord = features.first().geometry().getInfo()['coordinates']
    coord = np.array(coord).reshape(-1, 2) # Reshap in one array of tuple
    if len(coord) != 0:
      aoi.isEmpty = False
      aoi.yMin = round(min([y for x, y in coord]),precision)
      aoi.xMin = round(min([x for x, y in coord]),precision)
      aoi.yMax = round(max([y for x, y in coord]),precision)
      aoi.xMax = round(max([x for x, y in coord]),precision)
  return aoi

def saveImageFromGeeData(folderName, sat, aoi, aoiEE, date_start, date_end, log):
  offset = 100
  # lontal : 0.25 square at zoom=14
  refStep = 4096
  step = int(refStep*offset/(2**sat.zoom))
  count = 0
  
  x_min = int(aoi.xMin*offset)
  x_max = int(aoi.xMax*offset)
  y_min = int(aoi.yMin*offset)
  y_max = int(aoi.yMax*offset)
  
  data = sat.getDataWithArea(aoiEE, date_start, date_end).clip(aoiEE)
  url = data.getMapId(sat.vis)["tile_fetcher"].url_format
  
  folderName = 'img/'+folderName
  os.makedirs(folderName,exist_ok=True)
  relativeFolderName = folderName+'/'+sat.name
  
  for x in range(x_min, x_max, step):
    for y in range(y_min, y_max, step):
      count += 1
      
      east = x_max/offset if (x+step > x_max) else (x+step)/offset
      west = x/offset
      north = y_max/offset if (y+step > y_max) else (y+step)/offset
      south = y/offset
      
      while exists(relativeFolderName+'_'+str(count)+'.tif'):
        count += 1
      
      with log:
        log.clear_output()
        print('Part:',count,'w:',west,'e:',east,'s:',south,'n:',north)
      _, _ = contextily.bounds2raster(west,south,east,north,ll=True,path=relativeFolderName+'_'+str(count)+'.tif',source=url, zoom=sat.zoom)
      with log:
        print('Part:',count,'Done')

def displayArea(m, listSatellite, precision = 3):
  def on_save_btn_clicked(e):
    nonlocal currentLayer
    with output:
      aoiEE = ee.FeatureCollection(m.draw_last_feature)
      aoi = extractCoordFromFeatureCollection(aoiEE, precision)
      
      if (datepicker_start.value != None and datepicker_end.value != None 
        and datepicker_start.value < datepicker_end.value 
        and titleField.value != '' 
        and not aoi.isEmpty):
        
        if currentLayer != None:
          m.remove_ee_layer(currentLayer.name)

        date_start = datepicker_start.value.strftime('%Y-%m-%d')
        date_end = datepicker_end.value.strftime('%Y-%m-%d')
        
        sat = dropdown.value
        if sat != None:
          print(sat.name)
          currentLayer = sat.getDataWithArea(aoiEE, date_start, date_end).clip(aoiEE)
          m.addLayer(currentLayer, sat.vis, sat.name)
          saveImageFromGeeData(titleField.value, sat, aoi, aoiEE, date_start, date_end, log)
        else:
          print('no satellite selected!')

  def on_preview_btn_clicked(e):
    nonlocal currentLayer
    with output:
      aoiEE = ee.FeatureCollection(m.draw_features)
      aoi = extractCoordFromFeatureCollection(aoiEE, precision)
      print(aoi.isEmpty)
      if (datepicker_start.value != None and datepicker_end.value != None):
        if currentLayer != None:
          m.remove_ee_layer(currentLayer.name)
        date_start = datepicker_start.value.strftime('%Y-%m-%d')
        date_end = datepicker_end.value.strftime('%Y-%m-%d')
        sat = dropdown.value
        if sat != None:
          if aoi.isEmpty:
            currentLayer = sat.getDataWithDate(date_start, date_end)
          else:
            currentLayer = sat.getDataWithArea(aoiEE,date_start, date_end).clip(aoiEE)
          m.addLayer(currentLayer, sat.vis, sat.name)
  
  def show_coordinates(**args):
    latlon = args.get('coordinates')
    if args.get('type') == 'mousemove':
      with output_mouse_widget:
        output_mouse_widget.clear_output()
        print(latlon)
  
  currentLayer=None
  
  output_mouse_widget = widgets.Output()
  log = widgets.Output()
  
  datepicker_start = widgets.DatePicker(description='Start date',disabled=False)
  datepicker_end = widgets.DatePicker(description='End date',disabled=False)
  
  titleField = widgets.Text(
    value='',
    placeholder='Type the project title here',
    description='Folder:',
    disabled=False
  )
  previewBtn = widgets.Button(description="Preview !")
  saveBtn = widgets.Button(description="Save images !")
  output = widgets.Output()
  
  previewBtn.on_click(on_preview_btn_clicked)
  saveBtn.on_click(on_save_btn_clicked)
  
  dropdown = widgets.Dropdown(
    options=list(zip([sat.name for sat in listSatellite], listSatellite)), value=None, description='Satellites:'
  )
  
  m.on_interaction(show_coordinates)
  
  m.draw_features
  m.draw_last_feature
  
  m.add_control(ipyleaflet.WidgetControl(widget=saveBtn, position='bottomright'))
  m.add_control(ipyleaflet.WidgetControl(widget=previewBtn, position='bottomright'))
  m.add_control(ipyleaflet.WidgetControl(widget=dropdown, position='bottomright'))
  m.add_control(ipyleaflet.WidgetControl(widget=datepicker_end, position='bottomright'))
  m.add_control(ipyleaflet.WidgetControl(widget=datepicker_start, position='bottomright'))
  m.add_control(ipyleaflet.WidgetControl(widget=titleField, position='bottomright'))
  m.add_control(ipyleaflet.WidgetControl(widget=output_mouse_widget, position='bottomleft'))
  m.add_control(ipyleaflet.WidgetControl(widget=log, position='bottomleft'))

  display(m)
