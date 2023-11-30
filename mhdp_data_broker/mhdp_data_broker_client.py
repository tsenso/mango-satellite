import logging
from timeit import default_timer as timer

import grpc

# import the generated classes
from generated.data_broker_pb2 import Empty
from generated.data_broker_pb2_grpc import DataBrokerStub

port = 8061
max_msg_length = 1024 * 1024 * 200


def get_data_for_exact_year_prediction():
    logging.info("get_data_for_exact_year_prediction ...")

    with grpc.insecure_channel('localhost:{}'.format(port),
                               options=[('grpc.max_message_length', max_msg_length),
                                        ('grpc.max_send_message_length', max_msg_length),
                                        ('grpc.max_receive_message_length', max_msg_length)]) as channel:
        stub = DataBrokerStub(channel)
        response = stub.GetDataForExactYearPrediction(Empty())

    logging.info("Response received: ")
    logging.info(str(response))


def get_data_for_prediction():
    logging.info("get_data_for_prediction ...")

    with grpc.insecure_channel('localhost:{}'.format(port),
                               options=[('grpc.max_message_length', max_msg_length),
                                        ('grpc.max_send_message_length', max_msg_length),
                                        ('grpc.max_receive_message_length', max_msg_length)]) as channel:
        stub = DataBrokerStub(channel)
        response = stub.GetDataForPrediction(Empty())

    logging.info("Response received: ")
    logging.info(str(response))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # predict_images()
    get_data_for_exact_year_prediction()
    get_data_for_prediction()
