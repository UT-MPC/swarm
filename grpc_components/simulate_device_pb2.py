# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: simulate_device.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='simulate_device.proto',
  package='simulatedeivce',
  syntax='proto3',
  serialized_options=b'\n\037io.grpc.examples.simulatedeviceB\023SimulateDeviceProtoP\001\242\002\002SD',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x15simulate_device.proto\x12\x0esimulatedeivce\"\x07\n\x05\x45mpty\"\x18\n\x06\x43onfig\x12\x0e\n\x06\x63onfig\x18\x01 \x01(\t\"$\n\x06Status\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x0e\n\x06status\x18\x02 \x01(\t\"\x1d\n\x08WorkerId\x12\x11\n\tworker_id\x18\x01 \x01(\x05\x32\x8a\x03\n\x0eSimulateDevice\x12\x44\n\x0eSetWorkerState\x12\x18.simulatedeivce.WorkerId\x1a\x16.simulatedeivce.Status\"\x00\x12;\n\x07RunTask\x12\x16.simulatedeivce.Config\x1a\x16.simulatedeivce.Status\"\x00\x12<\n\x08StopTask\x12\x16.simulatedeivce.Config\x1a\x16.simulatedeivce.Status\"\x00\x12<\n\tGetStatus\x12\x15.simulatedeivce.Empty\x1a\x16.simulatedeivce.Status\"\x00\x12\x36\n\x04Ping\x12\x15.simulatedeivce.Empty\x1a\x15.simulatedeivce.Empty\"\x00\x12\x41\n\rSimulateOppCL\x12\x16.simulatedeivce.Config\x1a\x16.simulatedeivce.Status\"\x00\x42=\n\x1fio.grpc.examples.simulatedeviceB\x13SimulateDeviceProtoP\x01\xa2\x02\x02SDb\x06proto3'
)




_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='simulatedeivce.Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=41,
  serialized_end=48,
)


_CONFIG = _descriptor.Descriptor(
  name='Config',
  full_name='simulatedeivce.Config',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='config', full_name='simulatedeivce.Config.config', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=50,
  serialized_end=74,
)


_STATUS = _descriptor.Descriptor(
  name='Status',
  full_name='simulatedeivce.Status',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='simulatedeivce.Status.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='status', full_name='simulatedeivce.Status.status', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=76,
  serialized_end=112,
)


_WORKERID = _descriptor.Descriptor(
  name='WorkerId',
  full_name='simulatedeivce.WorkerId',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='worker_id', full_name='simulatedeivce.WorkerId.worker_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=114,
  serialized_end=143,
)

DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
DESCRIPTOR.message_types_by_name['Config'] = _CONFIG
DESCRIPTOR.message_types_by_name['Status'] = _STATUS
DESCRIPTOR.message_types_by_name['WorkerId'] = _WORKERID
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'simulate_device_pb2'
  # @@protoc_insertion_point(class_scope:simulatedeivce.Empty)
  })
_sym_db.RegisterMessage(Empty)

Config = _reflection.GeneratedProtocolMessageType('Config', (_message.Message,), {
  'DESCRIPTOR' : _CONFIG,
  '__module__' : 'simulate_device_pb2'
  # @@protoc_insertion_point(class_scope:simulatedeivce.Config)
  })
_sym_db.RegisterMessage(Config)

Status = _reflection.GeneratedProtocolMessageType('Status', (_message.Message,), {
  'DESCRIPTOR' : _STATUS,
  '__module__' : 'simulate_device_pb2'
  # @@protoc_insertion_point(class_scope:simulatedeivce.Status)
  })
_sym_db.RegisterMessage(Status)

WorkerId = _reflection.GeneratedProtocolMessageType('WorkerId', (_message.Message,), {
  'DESCRIPTOR' : _WORKERID,
  '__module__' : 'simulate_device_pb2'
  # @@protoc_insertion_point(class_scope:simulatedeivce.WorkerId)
  })
_sym_db.RegisterMessage(WorkerId)


DESCRIPTOR._options = None

_SIMULATEDEVICE = _descriptor.ServiceDescriptor(
  name='SimulateDevice',
  full_name='simulatedeivce.SimulateDevice',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=146,
  serialized_end=540,
  methods=[
  _descriptor.MethodDescriptor(
    name='SetWorkerState',
    full_name='simulatedeivce.SimulateDevice.SetWorkerState',
    index=0,
    containing_service=None,
    input_type=_WORKERID,
    output_type=_STATUS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='RunTask',
    full_name='simulatedeivce.SimulateDevice.RunTask',
    index=1,
    containing_service=None,
    input_type=_CONFIG,
    output_type=_STATUS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StopTask',
    full_name='simulatedeivce.SimulateDevice.StopTask',
    index=2,
    containing_service=None,
    input_type=_CONFIG,
    output_type=_STATUS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetStatus',
    full_name='simulatedeivce.SimulateDevice.GetStatus',
    index=3,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_STATUS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Ping',
    full_name='simulatedeivce.SimulateDevice.Ping',
    index=4,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SimulateOppCL',
    full_name='simulatedeivce.SimulateDevice.SimulateOppCL',
    index=5,
    containing_service=None,
    input_type=_CONFIG,
    output_type=_STATUS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_SIMULATEDEVICE)

DESCRIPTOR.services_by_name['SimulateDevice'] = _SIMULATEDEVICE

# @@protoc_insertion_point(module_scope)
