import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.polar_transform import (
    find_origin_angular_descent,
    find_origin_angular_grid,
    polar_transform,
)

from quantem.core.io.serialize import load
from pathlib import Path

# TODO: import the load function and generate a ds from sample data from Karen's tutorial
# TODO: use the sample ds to explore type(origin) (after calling angular descent function)

class PairAngleDistributionFunction(AutoSerialize):
    """
    Compute pair-angle distribution function for a given 4D-STEM dataset.

    Outline of extraction pipeline:
    - Create the PairAngleDistributionFunction class (just use init constructor)
        - Store dataset
        - Store diffraction-pattern center (if given)
        - Store other parameters from appendix B
    - Use the established function to find origin/center
    - Use polar transform
    - Rescale_intensity function (equation 7)
    - Compute angular cross-correlation - C(q, q’, delta phi) for ONE pattern, then average over ALL diffraction patterns
    - Decompose into spherical harmonic components
    - Spherical Bessel transform from q to r
    - Legendre sum to get padf!
    - Plot padf


    """
    def __init__(self,
                 ds: Dataset4dstem = None,
                 origin: NDArray | None = None
                 ):
        super().__init__()

        self.ds = ds
        self.origin = origin

        if origin is None:
            self.origin = find_origin_angular_grid(ds)

    def find_origin():
        pass







if __name__ == "__main__":
    path = Path(r"C:\Users\Emanu\Downloads\Ta_sim_binned.zip")
    ds = load(path)
    print(type(ds))

    padf = PairAngleDistributionFunction(ds=ds)
    print(padf.origin)
    print(type(padf.origin))