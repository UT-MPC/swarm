# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import simulate_device_pb2 as simulate__device__pb2


class SimulateDeviceStub(object):
    """Interface exported by the server.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.InitDevice = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/InitDevice',
                request_serializer=simulate__device__pb2.Config.SerializeToString,
                response_deserializer=simulate__device__pb2.Status.FromString,
                )
        self.StartOppCL = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/StartOppCL',
                request_serializer=simulate__device__pb2.Empty.SerializeToString,
                response_deserializer=simulate__device__pb2.Status.FromString,
                )
        self.CheckDevice = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/CheckDevice',
                request_serializer=simulate__device__pb2.Empty.SerializeToString,
                response_deserializer=simulate__device__pb2.Status.FromString,
                )
        self.GetResult = channel.unary_stream(
                '/simulatedeivce.SimulateDevice/GetResult',
                request_serializer=simulate__device__pb2.Empty.SerializeToString,
                response_deserializer=simulate__device__pb2.Result.FromString,
                )


class SimulateDeviceServicer(object):
    """Interface exported by the server.
    """

    def InitDevice(self, request, context):
        """A simple RPC.

        Initializes the Devices.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StartOppCL(self, request, context):
        """Starts the OppCL simulation.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckDevice(self, request, context):
        """Checks the Status of the Devices.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetResult(self, request, context):
        """Checks the Status of the Devices.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SimulateDeviceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'InitDevice': grpc.unary_unary_rpc_method_handler(
                    servicer.InitDevice,
                    request_deserializer=simulate__device__pb2.Config.FromString,
                    response_serializer=simulate__device__pb2.Status.SerializeToString,
            ),
            'StartOppCL': grpc.unary_unary_rpc_method_handler(
                    servicer.StartOppCL,
                    request_deserializer=simulate__device__pb2.Empty.FromString,
                    response_serializer=simulate__device__pb2.Status.SerializeToString,
            ),
            'CheckDevice': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckDevice,
                    request_deserializer=simulate__device__pb2.Empty.FromString,
                    response_serializer=simulate__device__pb2.Status.SerializeToString,
            ),
            'GetResult': grpc.unary_stream_rpc_method_handler(
                    servicer.GetResult,
                    request_deserializer=simulate__device__pb2.Empty.FromString,
                    response_serializer=simulate__device__pb2.Result.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'simulatedeivce.SimulateDevice', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class SimulateDevice(object):
    """Interface exported by the server.
    """

    @staticmethod
    def InitDevice(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/InitDevice',
            simulate__device__pb2.Config.SerializeToString,
            simulate__device__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StartOppCL(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/StartOppCL',
            simulate__device__pb2.Empty.SerializeToString,
            simulate__device__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CheckDevice(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/CheckDevice',
            simulate__device__pb2.Empty.SerializeToString,
            simulate__device__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetResult(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/simulatedeivce.SimulateDevice/GetResult',
            simulate__device__pb2.Empty.SerializeToString,
            simulate__device__pb2.Result.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
