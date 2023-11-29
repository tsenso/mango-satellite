from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

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

class Prediction(_message.Message):
    __slots__ = ("predicted_harvesting_year", "success", "predicted_harvesting_date_offset", "predicted_harvesting_date", "description")
    PREDICTED_HARVESTING_YEAR_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_HARVESTING_DATE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_HARVESTING_DATE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    predicted_harvesting_year: int
    success: bool
    predicted_harvesting_date_offset: int
    predicted_harvesting_date: _timestamp_pb2.Timestamp
    description: str
    def __init__(self, predicted_harvesting_year: _Optional[int] = ..., success: bool = ..., predicted_harvesting_date_offset: _Optional[int] = ..., predicted_harvesting_date: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., description: _Optional[str] = ...) -> None: ...

class Predictions(_message.Message):
    __slots__ = ("predictions",)
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    predictions: _containers.RepeatedCompositeFieldContainer[Prediction]
    def __init__(self, predictions: _Optional[_Iterable[_Union[Prediction, _Mapping]]] = ...) -> None: ...
