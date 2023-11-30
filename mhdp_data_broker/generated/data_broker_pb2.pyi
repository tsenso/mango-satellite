from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SatelliteImage(_message.Message):
    __slots__ = ("created_at", "tiff_image")
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    TIFF_IMAGE_FIELD_NUMBER: _ClassVar[int]
    created_at: _timestamp_pb2.Timestamp
    tiff_image: bytes
    def __init__(self, created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., tiff_image: _Optional[bytes] = ...) -> None: ...

class FeaturesForExactYear(_message.Message):
    __slots__ = ("satellite_images", "year_to_analyze")
    SATELLITE_IMAGES_FIELD_NUMBER: _ClassVar[int]
    YEAR_TO_ANALYZE_FIELD_NUMBER: _ClassVar[int]
    satellite_images: _containers.RepeatedCompositeFieldContainer[SatelliteImage]
    year_to_analyze: int
    def __init__(self, satellite_images: _Optional[_Iterable[_Union[SatelliteImage, _Mapping]]] = ..., year_to_analyze: _Optional[int] = ...) -> None: ...

class Features(_message.Message):
    __slots__ = ("satellite_images",)
    SATELLITE_IMAGES_FIELD_NUMBER: _ClassVar[int]
    satellite_images: _containers.RepeatedCompositeFieldContainer[SatelliteImage]
    def __init__(self, satellite_images: _Optional[_Iterable[_Union[SatelliteImage, _Mapping]]] = ...) -> None: ...
