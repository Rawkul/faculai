import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay
import scipy.spatial as sspat
import pandas as pd
from copy import deepcopy
from .utils import find_collinear_points

def compute_pixel_centroid(labeled_mask, label):
    """
    Computes the pixel coordinates of the centroid of a facula with the given
    label.
    
    Parameters
    ----------
    labeled_mask : ndarray
        Labeled mask (faculae) where the pixels corresponding to a facula with 
        a certain label (id) have all the same integer value (id).
    label : int
        The facula label (id or number).
        
    Returns
    -------
    center_clipped : ndarray
        The pixel coordinates of the centroid of the facula with the given label.
    """
    center = [np.average(indices) for indices in np.where(labeled_mask == label)]
    center_rounded = np.round(center).astype(int)
    center_clipped = np.clip(center_rounded, 0, labeled_mask.size - 1)
    return center_clipped

def compute_pixel_centroids(labeled_mask, num_faculae):
    """
    Computes the pixel coordinates of the centroids of all the faculae in the 
    labeled mask.
    
    Parameters
    ----------
    labeled_mask : ndarray
        Labeled mask (faculae) where the pixels corresponding to a facula with 
        a certain label (id) have all the same integer value (id).
    num_faculae : int
        The number of faculae in the labeled mask.
    
    Returns
    -------
    pole : ndarray
        The pixel coordinate of the pole of the map (i.e., the pixel coordinate 
        of the centroid of the largest facula).
    x : ndarray
        The pixel x-coordinates of the centroids of all the faculae in the 
        labeled mask.
    y : ndarray
        The pixel y-coordinates of the centroids of all the faculae in the 
        labeled mask.
    """
    centroids = [compute_pixel_centroid(labeled_mask, label) for label in range(1, num_faculae + 1)]
    centroids = np.array(centroids)
    pole, x, y = centroids[:, 0], centroids[:, 1], centroids[:,2]
    return pole, x, y

def spherical_to_cartesian(lat, lon):
    """
    Converts spherical coordinates (latitude and longitude) to cartesian coordinates.
    This function assumes a sphere of radius 1. If you want the output x, y, z
    to be located in a sphere of radius r, simply multiply x*r, y*r, z*r.
    
    Parameters
    ----------
    lat : float or ndarray
        The latitude(s) of the point(s) in degrees.
    lon : float or ndarray
        The longitude(s) of the point(s) in degrees.
        
    Returns
    -------
    x : float or ndarray
        The x-coordinate(s) of the point(s) in cartesian coordinates.
    y : float or ndarray
        The y-coordinate(s) of the point(s) in cartesian coordinates.
    z : float or ndarray
        The z-coordinate(s) of the point(s) in cartesian coordinates.
    """
    rlat = np.radians(lat)
    rlon = np.radians(lon)
    x = np.cos(rlat) * np.cos(rlon)
    y = np.cos(rlat) * np.sin(rlon)
    z = np.sin(rlat)
    return x, y, z

def get_mean_sd(labeled_mask, num_faculae, values):
    """
    Compute the mean and standard deviation of the provided values inside 
    each facula.
    
    Parameters
    ----------
    labeled_mask : numpy.ndarray
        Labeled mask (faculae) where the pixels corresponding to a facula with 
        a certain label (id) have all the same integer value (id).
    num_faculae : int
        Total number of faculae labels (ids) in the labeled mask.
    values : numpy.ndarray
        Values to be used in the mean and standard deviation computation.
        
    Returns
    -------
    mean : numpy.ndarray
        Mean value of "values" of each facula (label) in the labeled mask.
    sd : numpy.ndarray
        Standard deviation of "values" of each facula (label) in the labeled 
        mask.
    """
    all_labels = range(1, num_faculae + 1)
    mean = ndimage.mean(values, labeled_mask, all_labels)
    sd = ndimage.standard_deviation(values, labeled_mask, all_labels)
    return mean, sd

def get_number_of_pixels(labeled_mask, num_faculae):
    """
    Count the number of pixels for each facula label in the labeled mask.
    
    Parameters
    ----------
    labeled_mask : numpy.ndarray
        Labeled mask (faculae) where the pixels corresponding to a facula with 
        a certain label (id) have all the same integer value (id).
    num_faculae : int
        Total number of faculae labels (ids) in the labeled mask.
        
    Returns
    -------
    numpy.ndarray
        Number of pixels for each facula label in the labeled mask.
    """
    return ndimage.sum(labeled_mask > 0, labeled_mask, range(1, num_faculae + 1))

def get_spherical_distance(lat1, lon1, lat2, lon2, r):
    """
    Compute the distance between two points on a sphere using the 
    Haversine formula.
    
    Parameters
    ----------
    lat1 : float
        Latitude of point 1 in degrees.
    lon1 : float
        Longitude of point 1 in degrees.
    lat2 : float
        Latitude of point 2 in degrees.
    lon2 : float
        Longitude of point 2 in degrees.
    r : float
        Radius of the sphere.
        
    Returns
    -------
    float
        Distance between point 1 and point 2 on the sphere.
    """
    
    # Convert latitudes and longitudes to radians
    rlat1, rlon1, rlat2, rlon2 = np.radians([lat1, lon1, lat2, lon2])
    # Compute the differences in latitudes and longitudes
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    
    # Compute the haversine of half the angles between the points
    a = np.sin(dlat / 2.0)**2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0)**2
    
    # Compute the great-circle distance
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return c * r

# Se asume que los datos vienen quitados de nans
def get_facula_area(labeled_mask, label, lat, lon, pixel_size, r):
    """
    Compute the area of a facula with a given label.
    
    Parameters
    ----------
    labeled_mask : ndarray
        Labeled mask (faculae) where the pixels corresponding to a facula 
        with a certain label (id) have all the same integer value (id).
    label : int
        The facula label (id or number).
    lat : ndarray
        The latitude of each pixel.
    lon : ndarray
        The longitude of each pixel.
    pixel_size : float
        The size of each pixel in kilometers.
    r : float
        The radius of the Sun in kilometers.
        
    Returns
    -------
    area : float
        The area of the facula in square kilometers.
    """
    # Get specific facula area
    mask = labeled_mask == label
    # Apply mask to latitude and longitude
    mlat, mlon = lat[mask], lon[mask]
    # Get number of pixels for the facula:
    n = mask.sum()
    if n == 1:
        # In case there is a single pixel, the area will be the pixel size^2
        return pixel_size**2
    elif n == 2:
        # If there are 2 points, the shape is assumed as a rectangle
        # of sides pixel_size and distance between the points.
        distance = get_spherical_distance(mlat[0], mlon[0], mlat[1], mlon[1], r)
        return distance * pixel_size
    elif n == 3:
        x, y, z = spherical_to_cartesian(mlat, mlon)
        points = np.array([x, y, z]).T
        triangles = np.array([[0, 1, 2]], dtype = np.int32)
    else:
        # In other cases, do Delaunay triangulation on lat, lon space (r = cte)
        x, y, z = spherical_to_cartesian(mlat, mlon)
        points = np.array([x, y, z]).T
        # Triangulation is performed in a parametric 2D space to produce 
        # 3 vertice triangles. Since Rsun = cte, we can do this
        tri_points = np.array([mlat, mlon]).T
        
        # Check if any of the input points are collinear
        colinear = find_collinear_points(tri_points)
        if len(colinear) > 0:
          # Input points are collinear, add a small perturbation to break 
          # collinearity only in the collinear points
          tri_points[colinear] += np.random.randn(*tri_points[colinear].shape) * 1e-6
        try:
          # Compute Delaunay triangulation
          triangulation = Delaunay(tri_points)
        except:
          # An error occurred during triangulation, return negative area
          return -1.0
        triangles = triangulation.simplices
    
    # Compute the area
    area = 0
    for i in range(len(triangles)):
        a, b, c = points[triangles[i]]
        area += np.linalg.norm(np.cross(b-a, c-a)) * 0.5
       
    area *= r**2
    return area

def get_tables(model, input_data, compute_area = True):
    """
    Generate two tables with information related to polar faculae images:
    > Table 1 will get several features of all the faculae present in the input images.
    > Table 2 will get several features of the regions with no faculae in the input images.
    See below for more information.
    
    Parameters
    ----------
    model : object
        The trained machine learning model to use for faculae detection.
    compute_area: boolean
        If you wish to compute the area (true) with Delaunay triangulation or
        not (bool), in which case a value of 0 will be assigned for the area
        of each facula. Area ccomputation is slow, so you might want to avoid it.
    input_data : dict
        A dictionary containing the input data used for faculae detection. 
        Required keys are:
            - 'date': string, the date of the input data. Can be in any format,
              the output table will have the same date-time format.
            - 'ml': numpy array, the linear polarizarion of the image. This 
               data is used for faculae detection.
            - 'blos': numpy array, the magnetogram, i.e., the line-of-sight magnetic field image.
            - 'mi': numpy array, integrated I stokes parameter image.
            - 'lat': numpy array, the latitude coordinates of each pixel in the image, in degrees.
            - 'lon': numpy array, the longitude coordinates of each pixel in the image, in degrees.
            - 'x': numpy array, the x-coordinate of each pixel in arcseconds.
            - 'y': numpy array, the y-coordinate of each pixel in arcseconds.
            - 'pixel_size': float, the size of each pixel in arcseconds.
            - 'helio': numpy array, the heliocentric angles of each pixel in the image.
        Arrays can be of any size (2 X M x N) given both M >= 256 and N >= 256.
        In axis = 0, the index 0 is for North Pole data, and index = 0 is for 
        South Pole data. Also, it is assumed that the data is np.nan outsie the 
        Sun's observed sphere and below |+-latitudeº| values.
            
    Returns
    -------
    A tuple with 2 pandas DataFrames:
        
    #1 : pandas DataFrame,
        A table of faculae properties, with the following columns:
            - 'date': string, the date of the input data. This column serves as
               image id ince each image has a different observation date.
            - 'facula_id': int, the ID of each facula inside the image.
            - 'num_pixels': int, the number of pixels in each facula.
            - 'area': float, the area of each facula in square kilometers.
            - 'lat_centroid_mean': float, the mean latitude of each facula 
               in degrees.
            - 'lat_centroid_sd': float, the standard deviation of latitude of 
               each facula in degrees.
            - 'lon_centroid_mean': float, the mean longitude of each facula 
               in degrees.
            - 'lon_centroid_sd': float, the standard deviation of longitude of 
               each facula in degrees.
            - 'x_arcs_centroid_mean': float, the mean x-coordinate of each 
               facula in arcseconds.
            - 'x_arcs_centroid_sd': float, the standard deviation of x-coordinate 
               of each facula in arcseconds.
            - 'y_arcs_centroid_mean': float, the mean y-coordinate of each 
               facula in arcseconds.
            - 'y_arcs_centroid_sd': float, the standard deviation of y-coordinate 
               of each facula in arcseconds.
            - 'pole': string, the pole of the Sun where the faculae are located 
               (0 = North or 1 = South, is the axis 0 index value in the array)
            - 'x_pix_centroid': float, the x-coordinate of the centroid of each 
               facula in pixels (is the axis 1 index value in the array).
            - 'y_pix_centroid': float, the y-coordinate of the centroid of each 
               facula in pixels (is the axis 2 index value in the array).
            - 'blos_mean': float, the mean B_LOS value of each facula.
            - 'blos_sd': float, the standard deviation of B_LOS value of 
               each facula.
            - 'mi_mean': float, the mean I-magnetogram value of each facula.
            - 'mi_sd': float, the standard deviation of I-magnetogram value of 
               each facula.
            - 'ml_mean': float, the mean ml-magnetogram value of each facula.
            - 'ml_sd': float, the standard deviation of ml-magnetogram value of 
               each facula.
            - 'helio_mean': float, the mean heliocentric angle of each facula.
            - 'helio_sd': float, the standard deviation of the heliographic angles of each facula.
    
    #2 : pandas DataFrame,
        A table with information about the regions with no faculae:
            - 'date': string, the date of the input data. This column serves as
               image id since each image has a different observation date.
            - 'num_pixels_north/south': int, the number of pixels with no faculae in the
               North Pole/South Pole.
            - 'blos_mean_north/south': float, the mean B_LOS value in regions with no
               faculae in the North Pole/South Pole.
            - 'blos_sd_north/south': float, the standard deviation of B_LOS value in
               regions with no faculae in the North Pole/South Pole.
            - 'blos_max_north/south': float, the maximum B_LOS value in regions with no
               faculae in the North Pole/South Pole.
            - 'blos_min_north/south': float, the minimum B_LOS value in regions with no
               faculae in the North Pole/South Pole.
            - 'ml_mean_north/south': float, the mean mL value in regions with no
               faculae in the North Pole/South Pole.
            - 'ml_sd_north/south': float, the standard deviation of mL value in
               regions with no faculae in the North Pole/South Pole.
            - 'mi_mean_north/south': float, the mean I value in regions with no
               faculae in the North Pole/South Pole.
            - 'mi_sd_north/south': float, the standard deviation of I value in
               regions with no faculae in the North Pole/South Pole.
    """
    # Copy the data to modify it without affecting the original variable
    data = deepcopy(input_data)
    
    # Detect faculae
    faculae, num_faculae = model(data["ml"])
    inverse_mask = faculae.copy()
    
    # Compute centroids in pixel coordinates
    pole_c, x_px_c, y_px_c = compute_pixel_centroids(faculae, num_faculae)
    
    # Remove nans from the data
    nan_mask = ~np.isnan(data["ml"])
    faculae = faculae[nan_mask]
    data.pop("date")
    data.pop("pixel_size")
    for k in data.keys():
        data[k] = data[k][nan_mask]
    
    # Compute the centroids in latitude and longitude
    lat_centroids_mean, lat_centroids_sd = get_mean_sd(faculae, num_faculae, data["lat"])
    lon_centroids_mean, lon_centroids_sd = get_mean_sd(faculae, num_faculae, data["lon"])
    
    # Compute the centroids in x, y (arcsecs)
    x_centroids_mean, x_centroids_sd = get_mean_sd(faculae, num_faculae, data["x"])
    y_centroids_mean, y_centroids_sd = get_mean_sd(faculae, num_faculae, data["y"])
    
    # Compute the areas
    
    if compute_area:
        pixel_size = input_data["pixel_size"] # in km
        # Solar radius from https://iopscience.iop.org/article/10.3847/0004-6256/152/2/41
        r_sun = 695700 # km
        areas = [get_facula_area(faculae, facula, data["lat"], data["lon"], pixel_size, r_sun) for facula in range(1, num_faculae + 1)]
    else:
        areas = [0 for facula in range(1, num_faculae + 1)]

    # Number of pixels per facula
    num_px = get_number_of_pixels(faculae, num_faculae)
    
    # Stats for V-magnetogram
    b_mean, b_sd = get_mean_sd(faculae, num_faculae, data["blos"])
    
    # Stats for I-magnetogram
    mi_mean, mi_sd = get_mean_sd(faculae, num_faculae, data["mi"])
    
    # Stats for ml-magnetogram
    ml_mean, ml_sd = get_mean_sd(faculae, num_faculae, data["ml"])
    
    # Heliographic coordinates
    helio_mean, helio_sd = get_mean_sd(faculae, num_faculae, data["helio"])
    
    # Create the output table and add the data
    tbl = pd.DataFrame({"date" : input_data["date"],
                        "facula_id" : range(1, num_faculae + 1)})
    tbl["num_pixels"] = num_px
    tbl["area"] = areas
    tbl["lat_centroid_mean"] = lat_centroids_mean
    tbl["lat_centroid_sd"] = lat_centroids_sd
    tbl["lon_centroid_mean"] = lon_centroids_mean
    tbl["lon_centroid_sd"] = lon_centroids_sd
    tbl["x_arcs_centroid_mean"] = x_centroids_mean
    tbl["x_arcs_centroid_sd"] = x_centroids_sd
    tbl["y_arcs_centroid_mean"] = y_centroids_mean
    tbl["y_arcs_centroid_sd"] = y_centroids_sd
    tbl["pole"] = pole_c
    tbl["x_pix_centroid"] = x_px_c
    tbl["y_pix_centroid"] = y_px_c
    tbl["blos_mean"] = b_mean
    tbl["blos_sd"] = b_sd
    tbl["mi_mean"] = mi_mean
    tbl["mi_sd"] = mi_sd
    tbl["ml_mean"] = ml_mean
    tbl["ml_sd"] = ml_sd
    tbl["helio_mean"] = helio_mean
    tbl["helio_sd"] = helio_sd
    
    # Obtain magnetic field information from regions with no faculae.
    # Create an inverse mask
    inverse_mask = np.where((inverse_mask == 0) & ~np.isnan(input_data["ml"]), 1, 0).astype(np.uint8)

    # North Pole (1st position in array)
    num_px_n = np.sum(inverse_mask[0,:,:])
    b_mean_n = ndimage.mean(input_data["blos"][0,:,:], inverse_mask[0,:,:])
    b_max_n = ndimage.maximum(input_data["blos"][0,:,:], inverse_mask[0,:,:])
    b_min_n = ndimage.minimum(input_data["blos"][0,:,:], inverse_mask[0,:,:])
    b_sd_n = ndimage.standard_deviation(input_data["blos"][0,:,:], inverse_mask[0,:,:])
    
    # South Pole (2nd position in array)
    num_px_s = np.sum(inverse_mask[1,:,:])
    b_mean_s = ndimage.mean(input_data["blos"][1,:,:], inverse_mask[1,:,:])
    b_max_s = ndimage.maximum(input_data["blos"][1,:,:], inverse_mask[1,:,:])
    b_min_s = ndimage.minimum(input_data["blos"][1,:,:], inverse_mask[1,:,:])
    b_sd_s = ndimage.standard_deviation(input_data["blos"][1,:,:], inverse_mask[1,:,:])
    
    ml_mean_north = ndimage.mean(input_data["ml"][0,:,:], inverse_mask[0,:,:])
    ml_mean_south = ndimage.mean(input_data["ml"][1,:,:], inverse_mask[1,:,:])
    ml_sd_north = ndimage.standard_deviation(input_data["ml"][0,:,:], inverse_mask[0,:,:])
    ml_sd_south = ndimage.standard_deviation(input_data["ml"][1,:,:], inverse_mask[1,:,:])
    
    mi_mean_north = ndimage.mean(input_data["mi"][0,:,:], inverse_mask[0,:,:])
    mi_mean_south = ndimage.mean(input_data["mi"][1,:,:], inverse_mask[1,:,:])
    mi_sd_north = ndimage.standard_deviation(input_data["mi"][0,:,:], inverse_mask[0,:,:])
    mi_sd_south = ndimage.standard_deviation(input_data["mi"][1,:,:], inverse_mask[1,:,:])
    
    # Create another table with information for regions with no faculae.
    out_tbl = pd.DataFrame({"date" : [input_data["date"]],
                            "num_pixels_north" : [num_px_n],
                            "num_pixels_south" : [num_px_s],
                            "blos_mean_north" : [b_mean_n],
                            "blos_max_north" : [b_max_n],
                            "blos_min_north" : [b_min_n],
                            "blos_sd_north" : [b_sd_n],
                            "blos_mean_south" : [b_mean_s],
                            "blos_max_south" : [b_max_s],
                            "blos_min_south" : [b_min_s],
                            "blos_sd_south" : [b_sd_s],
                            "ml_mean_north" : [ml_mean_north],
                            "ml_sd_north" : [ml_sd_north],
                            "ml_mean_south" : [ml_mean_south],
                            "ml_sd_south" : [ml_sd_south],
                            "mi_mean_north" : [mi_mean_north],
                            "mi_sd_north" : [mi_sd_north],
                            "mi_mean_south" : [mi_mean_south],
                            "mi_sd_south" : [mi_sd_south]})
    
    return tbl, out_tbl

# Data comes with NaNs 
def get_cap_area(lat, lon, r_sun):
    """
    Parameters
    ----------
    lat : ndarray
        The latitude of each point in degrees.
    lon : ndarray
        The longitude of each point in degrees.
    r_sun : float
        The radius of the Sun in kilometers.

    Returns
    -------
    area : float
        The area of the spherical cap in square kilometers.
    """
    # Get only the surface points
    mask = ~np.isnan(lat)
    # Apply mask to latitude and longitude
    mlat, mlon = lat[mask], lon[mask]
    # Do Delaunay triangulation on lat, lon space (r = cte)
    x, y, z = spherical_to_cartesian(mlat, mlon)
    points = np.array([x, y, z]).T
    # Triangulation is performed in a parametric 2D space to produce 
    # 3 vertice triangles. Since Rsun = cte, we can do this
    tri_points = np.array([mlat, mlon]).T
    try:
      # Compute Delaunay triangulation
      triangulation = Delaunay(tri_points)
    except:
      # An error occurred during triangulation, return negative area
      return -1.0
    triangles = triangulation.simplices

    # Compute the area
    area = 0
    for i in range(len(triangles)):
        a, b, c = points[triangles[i]]
        area += np.linalg.norm(np.cross(b-a, c-a)) * 0.5
    area *= r_sun**2
    return area

def get_polar_areas(data):
    """
    Parameters
    ----------
    data : dictionary
        The data containing the date, latitude (lat) and longitude (lon) data
        for the north and south polar caps.

    Returns
    -------
    output : pandas.DataFrame
        A table with the following columns:
            - date: the date.
            - area_north: area of the north polar cap.
            - area_south: area of the south polar cap.
    """
    date = data["date"]
    lat = data["lat"]
    lon = data["lon"]
    # Solar radius from https://arxiv.org/abs/astro-ph/9803131
    r_sun = 695508 # km
    area_north = get_cap_area(lat[0,:,:], lon[0,:,:], r_sun)
    area_south = get_cap_area(lat[1,:,:], lon[1,:,:], r_sun)
    output = pd.DataFrame({"date" : [date], 
                           "area_north" : [area_north],
                           "area_south" : [area_south]})
    return output
