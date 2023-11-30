import logging
import os
from concurrent import futures
from datetime import datetime
from typing import List

import grpc

# import the generated classes :
from generated import data_broker_pb2_grpc
from generated.data_broker_pb2 import SatelliteImage, Features, FeaturesForExactYear

port = 8061


def load_images(path: str, filter_suffix: str = "") -> List[SatelliteImage]:
    images_filenames = os.listdir(path)

    if filter_suffix:
        images_filenames = list(filter(lambda filename: filename.endswith(filter_suffix), images_filenames))

    images = []

    for image_filename in images_filenames:
        image_filename_with_path = os.path.join(path, image_filename)

        try:
            created_at = datetime.strptime(image_filename.split("_")[0], "%Y%m%dT%H%M%S")
        except Exception as e:
            continue

        satellite_image = SatelliteImage()
        satellite_image.created_at.FromDatetime(created_at)
        with open(image_filename_with_path, "rb") as f:
            satellite_image.tiff_image = f.read()

        images.append(satellite_image)

    return images


# create a class to define the server functions, derived from
class DataBrokerServicer(data_broker_pb2_grpc.DataBrokerServicer):
    def __init__(self):
        self.has_metrics = True  # Flag to indicate the presence of metrics in this node and print a message accordingly.
        if self.has_metrics:
            logging.info('MetricsAvailable')

    def GetDataForPrediction(self, request, context):
        logging.debug(f"GetDataForPrediction: started data generation for request = {request}")

        response = Features()

        response.satellite_images.extend(load_images("assets/Pakistan_test/all", "N.tif"))

        return response

    def GetDataForExactYearPrediction(self, request, context):
        logging.debug(f"GetDataForPrediction: started data generation for request = {request}")

        response = FeaturesForExactYear(year_to_analyze=2018)

        response.satellite_images.extend(load_images("assets/Pakistan_test/2018_n"))

        return response


def serve():
    max_msg_length = 1024 * 1024 * 200
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_message_length', max_msg_length),
                                  ('grpc.max_send_message_length', max_msg_length),
                                  ('grpc.max_receive_message_length', max_msg_length)])

    data_broker_pb2_grpc.add_DataBrokerServicer_to_server(DataBrokerServicer(), server)

    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Server started, listening on {port}")

    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
