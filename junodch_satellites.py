import ee

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
    
class Landsat8(SatelliteData):
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
  def __init__(self, minRad=1, maxRad=2, name='Night VIIRS', zoom=9):
    self.name = name
    self.src = 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'
    self.vis = {
      'min': minRad,
      'max': maxRad, 
      'bands': ['avg_rad'], 
      'palette': ['010101','ffffff'],
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
  def __init__(self, name='Population GHSL', zoom=11):
    self.name = name
    self.src = 'JRC/GHSL/P2016/POP_GPW_GLOBE_V1'
    self.vis = {
      "min": 0.0,
      "max": 20.0,
      "palette": ['010101', 'ffffff'],
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
