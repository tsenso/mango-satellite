import logging
import os
from datetime import datetime
from typing import List, Dict

from model.harvest_predict import predict_real


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


class Image:
    created_at: datetime
    data: bytes

    def __init__(self, created_at, data):
        self.created_at = created_at
        self.data = data

    def _created_at_to_filename(self, extension: str = "tif", use_dummy_aaa_suffix: bool = True):
        timestamp_str = self.created_at.strftime("%Y%m%dT%H%M%S")
        suffix = "RVN" if use_dummy_aaa_suffix else ""
        return f"{timestamp_str}_{suffix}.{extension}"

    def store(self, path: str = "") -> str:
        filename = os.path.join(path, self._created_at_to_filename())
        with open(filename, 'wb') as f:
            f.write(self.data)

        return filename


def get_exact_date_of_border_time_for_year(year: int):
    # ToDo: This date will be hardcoded for now
    return datetime(year, 8, 1)


def images_separator_by_harvesting_year(images: List[Image]) -> Dict[int, List[Image]]:
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


def __predict_wrapper(images: List[Image]) -> (int, int):
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
        return datetime.strptime(last_date,"%Y-%m-%d").year, results[last_date]
    except Exception as e:
        raise ValueError("There is no enough data for prediction! Unable to predict!")


def predict_for_images_list(images: List[Image], planned_harvesting_year: int = -1) -> Dict[int, int]:
    if len(images) == 0:
        raise ValueError("There are no any image! Unable to predict!")

    images_by_harvesting_year = images_separator_by_harvesting_year(images)

    prediction_results = {}

    if planned_harvesting_year != -1:
        # We will predict only one year
        if planned_harvesting_year not in images_by_harvesting_year:
            raise ValueError(f"There are no images for {planned_harvesting_year} year! Unable to predict!")

        harvesting_year, offset = __predict_wrapper(images_by_harvesting_year[planned_harvesting_year])
        prediction_results[harvesting_year] = offset
    else:
        # We will predict as much as possible years
        for harvesting_year, images_for_year in images_by_harvesting_year.items():
            if not images_for_year or len(images_for_year) == 0:
                continue
            try:
                harvesting_year, offset = __predict_wrapper(images_for_year)
                prediction_results[harvesting_year] = offset
            except ValueError as e:
                logging.warning(f"Unable to predict {harvesting_year} harvesting year")

    return prediction_results
