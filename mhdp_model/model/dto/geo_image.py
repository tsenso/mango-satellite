import os
from datetime import datetime

from generated.model_pb2 import SatelliteImage


class GeoImage:
    created_at: datetime
    data: bytes

    def __init__(self, created_at, data):
        self.created_at = created_at
        self.data = data

    def _created_at_to_filename(self, extension: str = "tif", use_dummy_aaa_suffix: bool = True):
        timestamp_str = self.created_at.strftime("%Y%m%dT%H%M%S")
        suffix = "RVN" if use_dummy_aaa_suffix else ""
        return f"{timestamp_str}_{suffix}.{extension}"

    def store(self, path: str = "") -> str:
        filename = os.path.join(path, self._created_at_to_filename())
        with open(filename, 'wb') as f:
            f.write(self.data)

        return filename

    def map_to_grpc_model(self) -> SatelliteImage:
        grpc_satellite_image = SatelliteImage(tiff_image=self.data)
        grpc_satellite_image.created_at.FromDatetime(self.created_at)

        return grpc_satellite_image
