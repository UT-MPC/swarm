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
        self.SetWorkerInfo = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/SetWorkerInfo',
                request_serializer=simulate__device__pb2.WorkerInfo.SerializeToString,
                response_deserializer=simulate__device__pb2.Status.FromString,
                )
        self.RunTask = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/RunTask',
                request_serializer=simulate__device__pb2.Config.SerializeToString,
                response_deserializer=simulate__device__pb2.Status.FromString,
                )
        self.StopTask = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/StopTask',
                request_serializer=simulate__device__pb2.Config.SerializeToString,
                response_deserializer=simulate__device__pb2.Status.FromString,
                )
        self.GetStatus = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/GetStatus',
                request_serializer=simulate__device__pb2.Empty.SerializeToString,
                response_deserializer=simulate__device__pb2.Status.FromString,
                )
        self.ResetState = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/ResetState',
                request_serializer=simulate__device__pb2.Empty.SerializeToString,
                response_deserializer=simulate__device__pb2.Status.FromString,
                )
        self.CheckRunning = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/CheckRunning',
                request_serializer=simulate__device__pb2.Empty.SerializeToString,
                response_deserializer=simulate__device__pb2.Status.FromString,
                )
        self.CheckInitialized = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/CheckInitialized',
                request_serializer=simulate__device__pb2.Empty.SerializeToString,
                response_deserializer=simulate__device__pb2.Status.FromString,
                )
        self.Ping = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/Ping',
                request_serializer=simulate__device__pb2.Empty.SerializeToString,
                response_deserializer=simulate__device__pb2.Empty.FromString,
                )
        self.ClearCache = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/ClearCache',
                request_serializer=simulate__device__pb2.Empty.SerializeToString,
                response_deserializer=simulate__device__pb2.Empty.FromString,
                )
        self.SimulateOppCL = channel.unary_unary(
                '/simulatedeivce.SimulateDevice/SimulateOppCL',
                request_serializer=simulate__device__pb2.Config.SerializeToString,
                response_deserializer=simulate__device__pb2.Status.FromString,
                )


class SimulateDeviceServicer(object):
    """Interface exported by the server.
    """

    def SetWorkerInfo(self, request, context):
        """A simple RPC.

        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RunTask(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StopTask(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetStatus(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ResetState(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckRunning(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckInitialized(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Ping(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ClearCache(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SimulateOppCL(self, request, context):
        """Starts the OppCL simulation.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SimulateDeviceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SetWorkerInfo': grpc.unary_unary_rpc_method_handler(
                    servicer.SetWorkerInfo,
                    request_deserializer=simulate__device__pb2.WorkerInfo.FromString,
                    response_serializer=simulate__device__pb2.Status.SerializeToString,
            ),
            'RunTask': grpc.unary_unary_rpc_method_handler(
                    servicer.RunTask,
                    request_deserializer=simulate__device__pb2.Config.FromString,
                    response_serializer=simulate__device__pb2.Status.SerializeToString,
            ),
            'StopTask': grpc.unary_unary_rpc_method_handler(
                    servicer.StopTask,
                    request_deserializer=simulate__device__pb2.Config.FromString,
                    response_serializer=simulate__device__pb2.Status.SerializeToString,
            ),
            'GetStatus': grpc.unary_unary_rpc_method_handler(
                    servicer.GetStatus,
                    request_deserializer=simulate__device__pb2.Empty.FromString,
                    response_serializer=simulate__device__pb2.Status.SerializeToString,
            ),
            'ResetState': grpc.unary_unary_rpc_method_handler(
                    servicer.ResetState,
                    request_deserializer=simulate__device__pb2.Empty.FromString,
                    response_serializer=simulate__device__pb2.Status.SerializeToString,
            ),
            'CheckRunning': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckRunning,
                    request_deserializer=simulate__device__pb2.Empty.FromString,
                    response_serializer=simulate__device__pb2.Status.SerializeToString,
            ),
            'CheckInitialized': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckInitialized,
                    request_deserializer=simulate__device__pb2.Empty.FromString,
                    response_serializer=simulate__device__pb2.Status.SerializeToString,
            ),
            'Ping': grpc.unary_unary_rpc_method_handler(
                    servicer.Ping,
                    request_deserializer=simulate__device__pb2.Empty.FromString,
                    response_serializer=simulate__device__pb2.Empty.SerializeToString,
            ),
            'ClearCache': grpc.unary_unary_rpc_method_handler(
                    servicer.ClearCache,
                    request_deserializer=simulate__device__pb2.Empty.FromString,
                    response_serializer=simulate__device__pb2.Empty.SerializeToString,
            ),
            'SimulateOppCL': grpc.unary_unary_rpc_method_handler(
                    servicer.SimulateOppCL,
                    request_deserializer=simulate__device__pb2.Config.FromString,
                    response_serializer=simulate__device__pb2.Status.SerializeToString,
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
    def SetWorkerInfo(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/SetWorkerInfo',
            simulate__device__pb2.WorkerInfo.SerializeToString,
            simulate__device__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RunTask(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/RunTask',
            simulate__device__pb2.Config.SerializeToString,
            simulate__device__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StopTask(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/StopTask',
            simulate__device__pb2.Config.SerializeToString,
            simulate__device__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetStatus(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/GetStatus',
            simulate__device__pb2.Empty.SerializeToString,
            simulate__device__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ResetState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/ResetState',
            simulate__device__pb2.Empty.SerializeToString,
            simulate__device__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CheckRunning(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/CheckRunning',
            simulate__device__pb2.Empty.SerializeToString,
            simulate__device__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CheckInitialized(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/CheckInitialized',
            simulate__device__pb2.Empty.SerializeToString,
            simulate__device__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Ping(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/Ping',
            simulate__device__pb2.Empty.SerializeToString,
            simulate__device__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ClearCache(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/ClearCache',
            simulate__device__pb2.Empty.SerializeToString,
            simulate__device__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SimulateOppCL(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulatedeivce.SimulateDevice/SimulateOppCL',
            simulate__device__pb2.Config.SerializeToString,
            simulate__device__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
