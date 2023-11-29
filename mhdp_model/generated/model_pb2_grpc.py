# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import generated.model_pb2 as model__pb2


class PredictStub(object):
    """Define the service
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.PredictMangoHarvestingDateForYear = channel.unary_unary(
                '/Predict/PredictMangoHarvestingDateForYear',
                request_serializer=model__pb2.FeaturesForExactYear.SerializeToString,
                response_deserializer=model__pb2.Prediction.FromString,
                )
        self.PredictMangoHarvestingDates = channel.unary_unary(
                '/Predict/PredictMangoHarvestingDates',
                request_serializer=model__pb2.Features.SerializeToString,
                response_deserializer=model__pb2.Predictions.FromString,
                )


class PredictServicer(object):
    """Define the service
    """

    def PredictMangoHarvestingDateForYear(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PredictMangoHarvestingDates(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_PredictServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'PredictMangoHarvestingDateForYear': grpc.unary_unary_rpc_method_handler(
                    servicer.PredictMangoHarvestingDateForYear,
                    request_deserializer=model__pb2.FeaturesForExactYear.FromString,
                    response_serializer=model__pb2.Prediction.SerializeToString,
            ),
            'PredictMangoHarvestingDates': grpc.unary_unary_rpc_method_handler(
                    servicer.PredictMangoHarvestingDates,
                    request_deserializer=model__pb2.Features.FromString,
                    response_serializer=model__pb2.Predictions.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Predict', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Predict(object):
    """Define the service
    """

    @staticmethod
    def PredictMangoHarvestingDateForYear(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Predict/PredictMangoHarvestingDateForYear',
            model__pb2.FeaturesForExactYear.SerializeToString,
            model__pb2.Prediction.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PredictMangoHarvestingDates(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Predict/PredictMangoHarvestingDates',
            model__pb2.Features.SerializeToString,
            model__pb2.Predictions.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
