from collections.abc import Sequence
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.datastructures.dataset4d import Dataset4d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem




class DPC(AutoSerialize):
    """
    DPC reconstruciton class
    """


    _token = object()

    def __init__(
        self,
        dataset: Dataset4dstem,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use DriftCorrection.from_data() or .from_file() to instantiate this class."
            )
        self.dataset = dataset

    @classmethod
    def from_file(
        cls,
        file_path: Sequence[str],
        file_type: str | None = None,
    ) -> "DPC":
        dataset = Dataset4dstem.from_file(file_path, file_type=file_type)
        return cls.from_data(
            dataset,
        )

    @classmethod
    def from_data(
        cls,
        dataset: Union[Dataset4dstem, Dataset4d, NDArray],
    ) -> "DPC":

        return cls(
            dataset=dataset,
            _token=cls._token,
        )




    def preprocess(
        self,
    ):
        pass