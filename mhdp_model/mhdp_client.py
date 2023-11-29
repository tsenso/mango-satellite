import logging
import os
from datetime import datetime
from timeit import default_timer as timer
from typing import List

import grpc

# import the generated classes
from generated import model_pb2_grpc, model_pb2
from model.dto.geo_image import GeoImage

port = 8061


def load_images(path: str, filter_suffix: str = "") -> List[GeoImage]:
    # assets_path = "assets/Pakistan_test/all"
    images_filenames = os.listdir(path)

    if filter_suffix:
        images_filenames = list(filter(lambda filename: filename.endswith(filter_suffix), images_filenames))

    images = []

    for image_filename in images_filenames:
        image_filename_with_path = os.path.join(path, image_filename)
        created_at = datetime.strptime(image_filename.split("_")[0], "%Y%m%dT%H%M%S")
        with open(image_filename_with_path, "rb") as f:
            image_bytes = f.read()
            images.append(GeoImage(created_at, image_bytes))

    return images


def predict_images():
    logging.info("predict_images ...")
    start_ch = timer()

    max_msg_length = 1024 * 1024 * 200
    with grpc.insecure_channel('localhost:{}'.format(port),
                               options=[('grpc.max_message_length', max_msg_length),
                                        ('grpc.max_send_message_length', max_msg_length),
                                        ('grpc.max_receive_message_length', max_msg_length)]) as channel:
        stub = model_pb2_grpc.PredictStub(channel)
        request = model_pb2.Features()
        for image in load_images("assets/Pakistan_test/all", "N.tif"):
            request.satellite_images.append(image.map_to_grpc_model())
        logging.info("Calling MHDP_Stub.PredictMangoHarvestingDates ...")
        response = stub.PredictMangoHarvestingDates(request)

    logging.info("Greeter client received: ")
    logging.info(str(response))

    end_ch = timer()
    ch_time = end_ch - start_ch
    logging.debug('Time to query server = {}'.format(ch_time))
    return response


def predict_2018_year():
    logging.info("predict_2018_year ...")
    start_ch = timer()

    with grpc.insecure_channel('localhost:{}'.format(port)) as channel:
        stub = model_pb2_grpc.PredictStub(channel)
        request = model_pb2.FeaturesForExactYear(year_to_analyze=2018)
        for image in load_images("assets/Pakistan_test/2018_n"):
            request.satellite_images.append(image.map_to_grpc_model())

        logging.info("Calling MHDP_Stub.PredictMangoHarvestingDateForYear ...")
        response = stub.PredictMangoHarvestingDateForYear(request)

    logging.info("Response received: ")
    logging.info(str(response))

    end_ch = timer()
    ch_time = end_ch - start_ch
    logging.debug('Time to query server = {}'.format(ch_time))
    return response


def predict_2018_year_from_many_images():
    logging.info("predict_2018_year_from_many_images ...")
    start_ch = timer()

    with grpc.insecure_channel('localhost:{}'.format(port)) as channel:
        stub = model_pb2_grpc.PredictStub(channel)
        request = model_pb2.FeaturesForExactYear(year_to_analyze=2018)
        for image in load_images("assets/Pakistan_test/all", "N.tif"):
            request.satellite_images.append(image.map_to_grpc_model())

        logging.info("Calling MHDP_Stub.PredictMangoHarvestingDateForYear ...")
        response = stub.PredictMangoHarvestingDateForYear(request)

    logging.info("Response received: ")
    logging.info(str(response))

    end_ch = timer()
    ch_time = end_ch - start_ch
    logging.debug('Time to query server = {}'.format(ch_time))
    return response


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # predict_images()
    predict_2018_year()
    predict_2018_year_from_many_images()
