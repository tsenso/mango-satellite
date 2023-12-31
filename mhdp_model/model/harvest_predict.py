#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:35:04 2023

@author: alexanderstudt
"""
import logging
import math
import os
import pickle
import warnings
from datetime import datetime
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
import rasterio
from seglearn.feature_functions import all_features
from tsflex.features import MultipleFeatureDescriptors, FeatureCollection
from tsflex.features.integrations import seglearn_feature_dict_wrapper

warnings.filterwarnings("ignore")
def setup_prediction_loggers():
    logging.getLogger("feature_calculation_logger").setLevel(logging.ERROR)
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)
    logging.getLogger("rasterio.env").setLevel(logging.ERROR)
    logging.getLogger("rasterio._base").setLevel(logging.ERROR)
    logging.getLogger("rasterio._io").setLevel(logging.ERROR)
    logging.getLogger("rasterio._filepath").setLevel(logging.ERROR)

def predict_test():
    all_images_filenames = os.listdir("../mhdp_data_broker/assets/Pakistan_test/all")
    # input_data = list(filter(lambda image_filename: image_filename.endswith("RVN.tif"), all_images_filenames))
    input_data = all_images_filenames
    print(f"Total images: {len(input_data)}")
    result = predict_real(input_data, country="Pakistan", images_dir="../uploads")
    print(result)
    return result


def predict_real(images_names: List[str], country: str, images_dir: str = ""):
    setup_prediction_loggers()

    if country not in ["Australia", "Pakistan"]:
        print("Please enter either Australia or Pakistan")
        return None

    print("Load Images")
    images_dict_dict = load_images(images_names, images_dir)
    print("Feature Extraction")
    images_df = calc_means_and_cloud_cleaned_version(images_dict_dict)
    combis, combis_clean = calc_various_combs_(images_df)

    images_df_timelimit = {}
    # ToDo: why time limiting is present here???
    for name, df in images_df.items():
        images_df_timelimit[name] = df[df["Time"] < datetime.strptime("2022-01-01", '%Y-%m-%d').date()]
    australien, pakistan, indien = split_region(images_df_timelimit)
    sets = [australien, pakistan, indien]
    scalable_channels = [channel for channel in list(images_df_timelimit[list(images_df_timelimit)[0]].columns) if
                         channel[:7] == "Channel"]
    scaled = rescale(sets, scalable_channels, scale_sets=sets)
    relevant_channels = ["Clouds"] + ["Channel_{}_mean_clean".format(i) for i in range(1, 12)] + combis_clean
    reduced = remove_irelevant_features(scaled, relevant_channels)
    reduced = remove_more(reduced)

    print("Begin Feature Generation")
    featuresets, _ = feature_creation(reduced, relevant_channels)
    feature_scaled = second_rescale(featuresets, scale_sets=featuresets)
    feature_cleaned = cleaning(feature_scaled)
    # print(f"feature_cleaned[{len(feature_cleaned)}:{len(feature_cleaned[0])}]: {feature_cleaned}")
    all_same_or_nan_columns, s = column_classification(feature_cleaned[0][list(feature_cleaned[0])[0]])
    all_same = True
    for region in feature_cleaned:
        for name, df in region.items():
            if all_same_or_nan_columns == column_classification(df)[0] and s == column_classification(df)[1]:
                pass
            else:
                print(False)
                all_same = False

    if all_same:
        for i, region in enumerate(feature_cleaned):
            for name, df in region.items():
                feature_cleaned[i][name] = df[[col for col in df.columns if col not in all_same_or_nan_columns]]
    else:
        print("not all of them are the same. Check that!!!")

    feature_expanded = modify_features(feature_cleaned)
    regionsets = create_regionsets(feature_expanded)
    num_cols = [col for col in regionsets[0].columns if col != "Source_DF"]
    y_australien, y_pakistan, year_australien, year_pakistan = create_labels(regionsets)

    print("Start Prediction")

    if country == "Australia":
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_australien.pkl"), "rb") as file:
            clf = pickle.load(file)
    elif country == "Pakistan":
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_pakistan.pkl"), "rb") as file:
            clf = pickle.load(file)
    preds = clf.predict(regionsets[0][num_cols])
    farms = list(regionsets[0]["Source_DF"])
    summe = timedelta(days=0)
    summe = 0
    mse = 0
    pred = []
    true = []
    for i in range(y_australien.shape[0]):
        x = y_australien[i, 0]
        y = y_australien[i, 1]
        x_pred = preds[i, 0]
        y_pred = preds[i, 1]
        difference = inverse_y(x, y) - inverse_y(x_pred, y_pred)
        pred.append(inverse_y(x_pred, y_pred))
        true.append(inverse_y(x, y))
        days_difference = difference.total_seconds() / (24 * 60 * 60)
        summe += np.abs(days_difference)
        mse += days_difference ** 2
    df_scatter = pd.DataFrame({"True Date": true, "Predicted Date": pred, "True Year": year_australien, "Farm": farms})

    print("Finished")

    return convert_dict(df_scatter)

    # df_scatter_australien.to_csv(os.path.join(args.output_path, "predictions_australia.csv"))
    # df_scatter_pakistan.to_csv(os.path.join(args.output_path, "predictions_pakistan.csv"))


def load_images(files, image_path):
    """
    Loads satellite images from the specified image directory path.

    This function expects the 'image_path' to contain directories, each 
    representing a collection of satellite images captured at different 
    timestamps. It lists all directories in 'image_path' and processes 
    each directory individually. Each directory is expected to contain 
    geotiff files (.tif) representing satellite images. The files should 
    follow a naming convention like '20151216T060242_20151216T060303_T42RVN.tif', 
    where the name encodes the capture date, time, and other relevant metadata.

    The function processes each image file within these directories, reading 
    and stacking the bands from the geotiff files to create a multi-dimensional 
    array representation of the satellite data. These arrays are then stored in 
    a dictionary, with file names as keys, which is subsequently added to a main 
    dictionary keyed by the directory names.

    Parameter:
    image_path (str): Path to the main directory containing the folders of satellite image files.

    Returns:
    dict of dict: A nested dictionary where each key is a directory name and its value 
    is another dictionary. This inner dictionary's keys are file names and its values 
    are the corresponding multi-dimensional image data arrays.
    """

    # files = os.listdir(image_path)
    # Initialize a dictionary to hold image data for each file
    images_dict_dict = {}

    # Initialize a dictionary to hold band data for each image in the directory
    images_dict = {}

    # Loop through each path in the current image directory
    for path in files:

        # Create the full path to the current geotiff file
        geotiff_path = os.path.join(image_path, path)

        # Check if the current path is a .tif file
        if path.endswith("tif"):
            # Open the geotiff file using rasterio
            with rasterio.open(geotiff_path) as dataset:

                # Read each band from the geotiff file and stack them together
                for i in range(1, 17):
                    array = dataset.read(i)
                    if i == 1:
                        bands = array.copy()
                    else:
                        bands = np.dstack((bands, array))

                # Add the stacked bands to the images dictionary with the file name as the key
                images_dict[path] = bands

    # Add the images dictionary to the main dictionary with the file name as the key
    images_dict_dict["Australien"] = images_dict

    return images_dict_dict


def calc_means_and_cloud_cleaned_version(images_dict_dict):
    """
    Calculates mean values and creates cloud-cleaned versions for satellite image data.
    
    This function processes a nested dictionary of satellite images, where each 
    top-level key corresponds to a collection of images (e.g., from different dates 
    or locations). For each collection, it computes mean values of the image bands 
    and also generates a cloud-cleaned version of these means.
    
    The function iterates over each collection, sorting the images based on their 
    filenames (which encode information like dates), and then computes the mean 
    for each band. It also applies a custom method 'mean_with_clouds_zero' to 
    produce a cloud-cleaned mean value for each band, considering a cloud threshold. 
    The results are stored in a pandas DataFrame with columns for filename, date, 
    cloud percentage, and mean values for each band (both original and cloud-cleaned).
    
    The output is a dictionary where each key corresponds to one of the initial 
    keys in 'images_dict_dict', and the value is the DataFrame containing the 
    processed data for that image collection.
    
    Parameters:
    images_dict_dict (dict of dict): A nested dictionary where each top-level key 
    corresponds to a collection of images and its value is another dictionary 
    containing image data arrays keyed by filenames.
    
    Returns:
    dict: A dictionary where each key is a top-level key from 'images_dict_dict' 
    and its value is a pandas DataFrame containing the processed image data.
    """
    # Initialize a dictionary to hold the dataframes for each key in images_dict_dict
    images_df = {}

    # Iterate over each key-value pair in images_dict_dict
    for kidd, vidd in images_dict_dict.items():

        # Sort the dictionary based on the custom sort_key function
        images_dict_sorted = dict(sorted(vidd.items(), key=lambda item: sort_key(item[0])))

        # Initialize lists to store dates and means
        dates = []
        means = []
        means_clean = []
        cloud_percentage_list = []

        # Process each image in the sorted dictionary
        for k, v in images_dict_sorted.items():
            filename = os.path.basename(k)
            # Extract date from the filename and convert to a date object
            date_object = datetime.strptime(filename[:8], '%Y%m%d').date()
            dates.append(date_object)

            # Calculate the mean value for the image and add it to the list
            means.append(np.mean(v, axis=(0, 1)))

            rel_bands = np.zeros((12))
            for i in range(12):
                rel_bands[i], cloud_percentage = mean_with_clouds_zero(v[:, :, i], v[:, :, -1], 0.5)
            means_clean.append(rel_bands)
            cloud_percentage_list.append(cloud_percentage)

        # Create a new dataframe to store the processed data
        images = pd.DataFrame(columns=["Filename"])

        # Populate the dataframe with the filenames, NP values, time, and channel means
        images["Filename"] = list(images_dict_sorted.keys())
        NP = [filename[-5] for filename in list(images_dict_sorted.keys())]
        images["NP"] = NP
        images["Time"] = dates
        images["Clouds"] = cloud_percentage_list
        for i in range(16):
            means_channel = [m[i] for m in means]
            images["Channel_{}_mean".format(i + 1)] = means_channel
            if i < 12:
                means_channel = [m[i] for m in means_clean]
                images["Channel_{}_mean_clean".format(i + 1)] = means_channel

        # Store the dataframe in the main dictionary using the key from images_dict_dict
        images_df[kidd] = images

    return images_df


def sort_key(key):
    return key[:8] + key[9:15]


def mean_with_clouds_zero(channel, clouds, cloud_filter=0):
    not_cloud_pixel = 0
    mean = 0
    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            if clouds[i, j] == 0:
                mean += channel[i, j]
                not_cloud_pixel += 1
    cloud_percentage = 1 - not_cloud_pixel / channel.size

    if not_cloud_pixel / channel.size < cloud_filter:
        return np.nan, cloud_percentage
    return mean / not_cloud_pixel, cloud_percentage


def calc_various_combs_(images_df):
    for clean_str in ["", "_clean"]:
        for k, df in images_df.items():
            df["Vegetation_Index" + clean_str] = difcomb(df, "Channel_8_mean" + clean_str, "Channel_4_mean" + clean_str)
            df["Moisture_Index" + clean_str] = difcomb(df, "Channel_9_mean" + clean_str, "Channel_12_mean" + clean_str)
            df["NDVI1" + clean_str] = difcomb(df, "Channel_4_mean" + clean_str, "Channel_5_mean" + clean_str)
            df["NDVI2" + clean_str] = difcomb(df, "Channel_5_mean" + clean_str, "Channel_12_mean" + clean_str)

    combis = ["Vegetation_Index", "Moisture_Index", "NDVI1", "NDVI2"]
    combis_clean = [comb + "_clean" for comb in combis]
    return combis, combis_clean


def difcomb(df, ch1, ch2):
    return (1 + (df[ch1] - df[ch2]) / (df[ch1] + df[ch2])) / 2


def split_region(dfs_dict):
    australien = {}
    pakistan = {}
    indien = {}

    for name, df in dfs_dict.items():

        seperates = []
        for letter, nb in df["NP"].value_counts().items():
            if nb > 30:
                seperates.append((df[df["NP"] == letter], letter))

        if name.startswith("Australien"):
            for df_sep, letter in seperates:
                australien[name + "_" + letter] = df_sep

        elif name.startswith("Pakistan"):
            for df_sep, letter in seperates:
                pakistan[name + "_" + letter] = df_sep

        elif name.startswith("Indien"):
            for df_sep, letter in seperates:
                indien[name + "_" + letter] = df_sep

        else:
            print(f"Untreated {seperates}")

    return australien, pakistan, indien


def rescale(sets, scalable_channels, scale_sets=None, scale_value=None):
    if not scale_sets and not scale_value:
        print("choose at least one")
        return None
    if scale_sets and scale_value:
        print("Do not choose both")
        return None

    set_scaled = [{} for _ in range(3)]
    for i, region in enumerate(sets):
        for name, df in region.items():
            set_scaled[i][name] = df.copy()
            for channel in scalable_channels:
                if scale_sets:
                    set_scaled[i][name][channel] = df[channel] / scale_sets[i][name][channel].max()
                else:
                    set_scaled[i][name][channel] = df[channel] / scale_value

    return set_scaled


def remove_irelevant_features(sets_scaled, relevant_channels):
    sets_reduced = [{} for _ in range(3)]
    for i, region in enumerate(sets_scaled):
        for name, df in region.items():
            sets_reduced[i][name] = df[["Time"] + relevant_channels].copy()
            sets_reduced[i][name] = sets_reduced[i][name].set_index("Time")
            sets_reduced[i][name].index = pd.to_datetime(sets_reduced[i][name].index)
    return sets_reduced


def remove_more(sets):
    sets_removed = [{} for _ in range(3)]
    for i, region in enumerate(sets):
        for name, df in region.items():
            index_list = []
            filtered_rows = []
            for index, row in df.iterrows():
                if index not in index_list:
                    filtered_rows.append(row)
                    index_list.append(index)
            sets_removed[i][name] = pd.DataFrame(filtered_rows, columns=df.columns)
            sets_removed[i][name] = sets_removed[i][name][sets_removed[i][name]["Clouds"] == 0]
            sets_removed[i][name] = sets_removed[i][name].dropna()
    return sets_removed


def feature_creation(sets, relevant_channels):
    sciped = []
    featuresets = [{} for _ in range(3)]

    fc = FeatureCollection(
        MultipleFeatureDescriptors(
            functions=seglearn_feature_dict_wrapper(all_features()),
            series_names=relevant_channels,
            windows=["90days"],
            strides="15days",
        )
    )

    for i, region in enumerate(sets):
        for name, df in region.items():
            # print(name)
            try:
                fdf = fc.calculate(data=df[df["Clouds"] == 0], approve_sparsity=True, return_df=True)
                featuresets[i][name] = fdf
            except:
                sciped.append(name)

    return featuresets, sciped


def second_rescale(sets, scale_sets=None, scale_value=None):
    if not scale_sets and not scale_value:
        print("choose at least one")
        return None
    if scale_sets and scale_value:
        print("Do not choose both")
        return None

    featuresets_scaled = [{} for _ in range(3)]

    for i, region in enumerate(sets):
        for name, df in region.items():
            featuresets_scaled[i][name] = df.copy()
            for channel in df.columns:
                featuresets_scaled[i][name][channel] = featuresets_scaled[i][name][channel] / scale_sets[i][name][
                    channel].max()
    return featuresets_scaled


def cleaning(sets):
    dropcols = ["Clouds__kurt__w=90D", "Clouds__skew__w=90D", "Clouds__variation__w=90D"]
    dropframe = ["Australien_3_M", "Indien_1_J", "Indien_3_N", "Indien_3_M", "Indien_2_G"]

    featuresets_cleaned = [{}, {}]
    for j, region in enumerate(sets):
        if j < 2:
            for name, df in region.items():
                # print(name, df.shape)
                if name not in dropframe:
                    featuresets_cleaned[j][name] = df.drop(dropcols, axis=1)
    return featuresets_cleaned


def column_classification(df):
    all_same_or_nan_columns = []
    nan_but_not_all_columns = []

    for col in df.columns:
        # Check if all values in the column are the same or all are NaN
        if df[col].nunique() == 1 or df[col].isna().all():
            all_same_or_nan_columns.append(col)
        # Check if there are any NaN values, but not all values are NaN
        elif df[col].isna().any():
            nan_but_not_all_columns.append(col)

    return all_same_or_nan_columns, nan_but_not_all_columns


def modify_features(sets):
    feature_expanded = [{}, {}]
    for j, region in enumerate(sets):
        # print(j)
        for name, df in region.items():
            df_expanded = pd.DataFrame(index=df.index)
            for col in df.columns:
                df_expanded[col] = df[col]

                for i in range(1, 6):  # for lags from 1 to 5
                    df_expanded[f'{col}_lag_{i}'] = df[col].shift(i)

            df_expanded = df_expanded.dropna()

            feature_expanded[j][name] = df_expanded

    return feature_expanded


def create_regionsets(sets):
    regionsets = []
    cols = sets[0][list(sets[0])[0]].columns
    for i, region in enumerate(sets):
        df_stack = pd.DataFrame(columns=cols)
        counter = 0
        letters = ["a", "b", "c", "d", "e", "f", "g"]
        for name, df in region.items():
            # Create a temporary copy of the dataframe to avoid modifying the original
            temp_df = df.copy()
            # Add a new column with the name of the dataframe
            temp_df['Source_DF'] = "Farm_{}".format(letters[counter])
            # Stack the dataframe
            df_stack = pd.concat([df_stack, temp_df])
            counter += 1
        regionsets.append(df_stack)
    return regionsets


def date_to_degree(date):
    # Determine the day of the year
    day_of_year = date.timetuple().tm_yday

    # Determine the total number of days in the year
    if date.year % 4 == 0:
        if date.year % 100 != 0 or (date.year % 100 == 0 and date.year % 400 == 0):
            total_days = 366
        else:
            total_days = 365
    else:
        total_days = 365

    # Convert the day to degrees
    degree = (day_of_year / total_days) * 360
    return degree


def map_degree_to_circle_point(degree):
    # Convert degree to radians and adjust for your circle's starting point
    theta = math.radians(90 - degree)

    # Circle parameters
    r = 0.5
    h = 0.5
    k = 0.5

    # Calculate coordinates
    x = r * math.cos(theta) + h
    y = r * math.sin(theta) + k

    return x, y


def calc_y(date):
    return map_degree_to_circle_point(date_to_degree(date))


def map_point_to_degree(x, y):
    # Circle parameters
    h = 0.5
    k = 0.5

    # Compute theta using arctan2
    theta = math.atan2(y - k, x - h)

    # Convert radians to degrees
    degree = math.degrees(theta)

    # Adjust the degree as per your circle's starting point
    adjusted_degree = 90 - degree
    if adjusted_degree < 0:
        adjusted_degree += 360

    return adjusted_degree


def degree_to_date(degree, year=2023):
    # Determine the total number of days in the year
    if year % 4 == 0:
        if year % 100 != 0 or (year % 100 == 0 and year % 400 == 0):
            total_days = 366
        else:
            total_days = 365
    else:
        total_days = 365

    # Convert the degree to a day of the year
    day_of_year = round((degree / 360) * total_days)

    # Get the date
    date = datetime(year, 1, 1) + timedelta(day_of_year - 1)
    return date


def inverse_y(x, y, year=2023):
    return degree_to_date(map_point_to_degree(x, y), year)


def create_labels(regionsets):
    y_australien = np.zeros((regionsets[0].shape[0], 2))
    y_pakistan = np.zeros((regionsets[1].shape[0], 2))
    year_australien = []
    year_pakistan = []
    for i, y_train in enumerate([y_australien, y_pakistan]):
        for j in range(y_train.shape[0]):
            x, y = calc_y(regionsets[i].index[j])
            y_train[j, 0] = x
            y_train[j, 1] = y
            if i == 0:
                year_australien.append(regionsets[0].index.year[j])
            elif i == 1:
                year_pakistan.append(regionsets[1].index.year[j])

    return y_australien, y_pakistan, year_australien, year_pakistan


def convert_dict(df_scatter):
    date_harvest = {}
    for index, row in df_scatter.iterrows():

        key = row["True Date"].strftime('%m-%d')
        key = str(row["True Year"]) + "-" + key

        date_harvest[key] = days_difference_ignoring_year(row["True Date"], row["Predicted Date"])
        # if index < 10:
        logging.debug("Predicted offset for %s: %d", key, date_harvest[key])
    return date_harvest


def days_difference_ignoring_year(timestamp1, timestamp2):
    """
    Calculate the difference in days between two dates, ignoring the year.
    The sign indicates which date is ahead. A negative value means timestamp1 is ahead of timestamp2.
    """
    # Extract month and day from each timestamp
    month_day1 = timestamp1.strftime('%m-%d')
    month_day2 = timestamp2.strftime('%m-%d')

    # Convert month and day into datetime objects with a fixed year (e.g., 2000)
    fixed_year = 2000
    date1 = datetime.strptime(f'{fixed_year}-{month_day1}', '%Y-%m-%d')
    date2 = datetime.strptime(f'{fixed_year}-{month_day2}', '%Y-%m-%d')

    # Calculate the difference in days
    day_difference = (date1 - date2).days

    # Adjust the difference if it is larger than half a year, considering the shortest path around the calendar
    if day_difference > 182:  # More than half a year ahead
        day_difference -= 365
    elif day_difference < -182:  # More than half a year behind
        day_difference += 365

    return -day_difference


def create_new_dictionary_with_date_offsets(original_dict, harvest_date):
    """
    Generate a new dictionary with the same keys as the original dictionary,
    but the values are the given date plus the value of the original dictionary corresponding to the key.
    """
    new_dict = {}

    # Convert the specific date to a datetime object
    specific_date_obj = datetime.strptime(harvest_date, '%Y-%m-%d')

    for key, value in original_dict.items():
        # Add the integer value (days) to the specific date
        new_date = specific_date_obj + timedelta(days=value)

        # Format the new date back to string and assign to the new dictionary
        new_dict[key] = new_date.strftime('%Y-%m-%d')

    return new_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    predict_test()
    # input_data = os.listdir(os.path.join("Images", "Pakistan_1"))
    # print(predict(input_data, "Pakistan", image_dir=os.path.join("Images", "Pakistan_1"), planned_Harvesting_Date="2016-05-30"))
