
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .dicom import load_image
from .endpoint import extract_label, image_to_record, predict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DICOM inference with Databricks."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to a DICOM file",
    )
    args = parser.parse_args()

    load_dotenv()

    if not os.environ.get("DATABRICKS_ENDPOINT"):
        parser.error("Configure DATABRICKS_ENDPOINT in .env.")

    image = load_image(args.image)
    record = image_to_record(image)
    response = predict(record)

    print(extract_label(response))


if __name__ == "__main__":
    main()
