# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import data_broker_pb2 as data__broker__pb2


class DataBrokerStub(object):
    """Define the service
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetDataForExactYearPrediction = channel.unary_unary(
                '/DataBroker/GetDataForExactYearPrediction',
                request_serializer=data__broker__pb2.Empty.SerializeToString,
                response_deserializer=data__broker__pb2.FeaturesForExactYear.FromString,
                )
        self.GetDataForPrediction = channel.unary_unary(
                '/DataBroker/GetDataForPrediction',
                request_serializer=data__broker__pb2.Empty.SerializeToString,
                response_deserializer=data__broker__pb2.Features.FromString,
                )


class DataBrokerServicer(object):
    """Define the service
    """

    def GetDataForExactYearPrediction(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDataForPrediction(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DataBrokerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetDataForExactYearPrediction': grpc.unary_unary_rpc_method_handler(
                    servicer.GetDataForExactYearPrediction,
                    request_deserializer=data__broker__pb2.Empty.FromString,
                    response_serializer=data__broker__pb2.FeaturesForExactYear.SerializeToString,
            ),
            'GetDataForPrediction': grpc.unary_unary_rpc_method_handler(
                    servicer.GetDataForPrediction,
                    request_deserializer=data__broker__pb2.Empty.FromString,
                    response_serializer=data__broker__pb2.Features.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'DataBroker', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class DataBroker(object):
    """Define the service
    """

    @staticmethod
    def GetDataForExactYearPrediction(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataBroker/GetDataForExactYearPrediction',
            data__broker__pb2.Empty.SerializeToString,
            data__broker__pb2.FeaturesForExactYear.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetDataForPrediction(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataBroker/GetDataForPrediction',
            data__broker__pb2.Empty.SerializeToString,
            data__broker__pb2.Features.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
