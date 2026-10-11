
from pathlib import Path

import numpy as np
import pydicom
import torch


def load_image(path: str | Path) -> torch.Tensor:
    """Load a single 2D grayscale DICOM as a normalized tensor.

    Returns a tensor with shape [1, height, width].
    """
    dataset = pydicom.dcmread(str(path))
    pixels = dataset.pixel_array.astype(np.float32)

    if pixels.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, got {pixels.shape}"
        )

    # Basic normalization for this starter project.
    pixels -= pixels.min()
    maximum = pixels.max()

    if maximum > 0:
        pixels /= maximum

    return torch.from_numpy(
        np.ascontiguousarray(pixels)
    ).unsqueeze(0)
