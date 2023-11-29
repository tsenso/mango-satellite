import logging
import os
from datetime import datetime
from typing import List, Dict

from model.dto.geo_image import GeoImage
from model.harvest_predict import predict_real
from model.dto.mango_harvesting_prediction import MangoHarvestingPrediction


def cleanup_dir(path: str):
    if path == '/' or path == "\\":
        return

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def verify_dir_created(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def get_exact_date_of_border_time_for_year(year: int):
    # ToDo: This date will be hardcoded for now
    return datetime(year, 8, 1)


def images_separator_by_harvesting_year(images: List[GeoImage]) -> Dict[int, List[GeoImage]]:
    if not images or len(images) == 0:
        return {}

    years = set(map(lambda image: image.created_at.year, images))
    harvesting_periods_borders = map(lambda year: get_exact_date_of_border_time_for_year(year),
                                     range(min(years), max(years) + 1 + 1))

    images_by_harvesting_year = {}

    previous_harvesting_border = None
    for harvesting_periods_border in harvesting_periods_borders:
        images_for_current_harvesting_year = list(filter(lambda
                                                             image: previous_harvesting_border <= image.created_at < harvesting_periods_border if previous_harvesting_border else image.created_at < harvesting_periods_border,
                                                         images))
        if len(images_for_current_harvesting_year):
            images_by_harvesting_year[harvesting_periods_border.year] = images_for_current_harvesting_year

        previous_harvesting_border = harvesting_periods_border

    return images_by_harvesting_year


def __predict_wrapper(images: List[GeoImage]) -> (int, int):
    path = ".temp"
    verify_dir_created(path)
    cleanup_dir(path)

    images_filenames_to_predict = list(map(lambda image: image.store(path), images))

    # ToDo: For now it works only for pakistan
    try:
        results = predict_real(images_filenames_to_predict, "Pakistan")
        if not results or len(results) == 0:
            raise ValueError("Got empty prediction from model!")

        last_date = list(results)[-1]
        return datetime.strptime(last_date, "%Y-%m-%d").year, results[last_date]
    except Exception as e:
        raise ValueError("There is no enough data for prediction! Unable to predict!")


def predict_for_images_list(images: List[GeoImage], planned_harvesting_year: int = -1) -> Dict[
    int, MangoHarvestingPrediction]:
    if len(images) == 0:
        raise ValueError("There are no any image! Unable to predict!")

    images_by_harvesting_year = images_separator_by_harvesting_year(images)

    prediction_results = {}

    if planned_harvesting_year != -1:
        # We will predict only one year
        if planned_harvesting_year not in images_by_harvesting_year:
            message = f"There are no images for {planned_harvesting_year} year! Unable to predict!"
            logging.warning(message)
            prediction_results[planned_harvesting_year] = MangoHarvestingPrediction(planned_harvesting_year,
                                                                                    description=message)

        try:
            last_image_year_used_for_prediction, offset = __predict_wrapper(
                images_by_harvesting_year[planned_harvesting_year])
            prediction_results[planned_harvesting_year] = MangoHarvestingPrediction(planned_harvesting_year,
                                                                                    success=True,
                                                                                    predicted_harvesting_date_offset=offset)
        except ValueError as e:
            message = f"Unable to predict {planned_harvesting_year} harvesting year"
            logging.warning(message)
            prediction_results[planned_harvesting_year] = MangoHarvestingPrediction(planned_harvesting_year,
                                                                                    description=message)
    else:
        # We will predict as much as possible years
        for harvesting_year, images_for_year in images_by_harvesting_year.items():
            if not images_for_year or len(images_for_year) == 0:
                continue
            try:
                last_image_year_used_for_prediction, offset = __predict_wrapper(images_for_year)
                prediction_results[harvesting_year] = MangoHarvestingPrediction(harvesting_year, success=True,
                                                                                predicted_harvesting_date_offset=offset)
            except ValueError as e:
                message = f"Unable to predict {harvesting_year} harvesting year"
                logging.warning(message)
                prediction_results[harvesting_year] = MangoHarvestingPrediction(harvesting_year,
                                                                                description=message)

    return prediction_results
