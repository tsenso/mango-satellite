import logging
from concurrent import futures

import grpc

# import the generated classes :
from generated import model_pb2
from generated import model_pb2_grpc
from model.model_adapter import predict_for_images_list
from model.dto.geo_image import GeoImage

port = 8061


# create a class to define the server functions, derived from
class PredictServicer(model_pb2_grpc.PredictServicer):
    def __init__(self):
        self.has_metrics = True  # Flag to indicate the presence of metrics in this node and print a message accordingly.
        if self.has_metrics:
            logging.info('MetricsAvailable')

    def PredictMangoHarvestingDates(self, request, context):
        logging.debug(f"PredictMangoHarvestingDates: started prediction for request = {request}")

        images = []
        for satellite_image in request.satellite_images:
            images.append(GeoImage(satellite_image.created_at.ToDatetime(),
                                   satellite_image.tiff_image))
        results = predict_for_images_list(images)

        response = model_pb2.Predictions()
        for year, prediction in results.items():
            response.predictions.append(prediction.map_to_grpc_model())
        return response

    def PredictMangoHarvestingDateForYear(self, request, context):
        logging.debug(f"PredictMangoHarvestingDateForYear: started prediction for request = {request}")

        images = []
        for satellite_image in request.satellite_images:
            images.append(GeoImage(satellite_image.created_at.ToDatetime(),
                                   satellite_image.tiff_image))

        results = predict_for_images_list(images, request.year_to_analyze)

        response = results[list(results)[-1]].map_to_grpc_model()
        return response


def serve():
    max_msg_length = 1024 * 1024 * 200
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                               options=[('grpc.max_message_length', max_msg_length),
                                        ('grpc.max_send_message_length', max_msg_length),
                                        ('grpc.max_receive_message_length', max_msg_length)])

    model_pb2_grpc.add_PredictServicer_to_server(PredictServicer(), server)

    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Server started, listening on {port}")

    # threading.Thread(target=app_run()).start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
