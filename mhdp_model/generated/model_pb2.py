# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: model.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0bmodel.proto\x1a\x1fgoogle/protobuf/timestamp.proto\"T\n\x0eSatelliteImage\x12.\n\ncreated_at\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x12\n\ntiff_image\x18\x02 \x01(\x0c\"Z\n\x14\x46\x65\x61turesForExactYear\x12)\n\x10satellite_images\x18\x01 \x03(\x0b\x32\x0f.SatelliteImage\x12\x17\n\x0fyear_to_analyze\x18\x02 \x01(\x05\"5\n\x08\x46\x65\x61tures\x12)\n\x10satellite_images\x18\x01 \x03(\x0b\x32\x0f.SatelliteImage\"\xf6\x01\n\nPrediction\x12!\n\x19predicted_harvesting_year\x18\x01 \x01(\x05\x12\x0f\n\x07success\x18\x02 \x01(\x08\x12(\n predicted_harvesting_date_offset\x18\x03 \x01(\x05\x12\x42\n\x19predicted_harvesting_date\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.TimestampH\x00\x88\x01\x01\x12\x18\n\x0b\x64\x65scription\x18\x05 \x01(\tH\x01\x88\x01\x01\x42\x1c\n\x1a_predicted_harvesting_dateB\x0e\n\x0c_description\"/\n\x0bPredictions\x12 \n\x0bpredictions\x18\x01 \x03(\x0b\x32\x0b.Prediction2\x8a\x01\n\x07Predict\x12G\n!PredictMangoHarvestingDateForYear\x12\x15.FeaturesForExactYear\x1a\x0b.Prediction\x12\x36\n\x1bPredictMangoHarvestingDates\x12\t.Features\x1a\x0c.Predictionsb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'model_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_SATELLITEIMAGE']._serialized_start=48
  _globals['_SATELLITEIMAGE']._serialized_end=132
  _globals['_FEATURESFOREXACTYEAR']._serialized_start=134
  _globals['_FEATURESFOREXACTYEAR']._serialized_end=224
  _globals['_FEATURES']._serialized_start=226
  _globals['_FEATURES']._serialized_end=279
  _globals['_PREDICTION']._serialized_start=282
  _globals['_PREDICTION']._serialized_end=528
  _globals['_PREDICTIONS']._serialized_start=530
  _globals['_PREDICTIONS']._serialized_end=577
  _globals['_PREDICT']._serialized_start=580
  _globals['_PREDICT']._serialized_end=718
# @@protoc_insertion_point(module_scope)
