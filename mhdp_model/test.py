import logging
import os
from datetime import datetime

from model.model_adapter import predict_for_images_list
from model.dto.geo_image import GeoImage

logging.basicConfig(level=logging.DEBUG)


# /c/Program\ Files\ \(x86\)/GnuWin32/bin/tiffinfo.exe 20170802T055639_AAA.tif
def main():
    assets_path = "assets/Pakistan_test/all"
    all_images_filenames = os.listdir(assets_path)
    # images_filenames = all_images_filenames
    images_filenames = list(filter(lambda image_filename: image_filename.endswith("RVN.tif"), all_images_filenames))

    images = []

    for image_filename in images_filenames:
        filename = os.path.join(assets_path, image_filename)
        created_at = datetime.strptime(image_filename.split("_")[0], "%Y%m%dT%H%M%S")
        with open(filename, "rb") as f:
            image_bytes = f.read()
            images.append(GeoImage(created_at, image_bytes))

    print(f"Total images loaded: {len(images)}")
    # results = __predict_wrapper(images)
    results = predict_for_images_list(images)
    print(results)


if __name__ == "__main__":
    main()
