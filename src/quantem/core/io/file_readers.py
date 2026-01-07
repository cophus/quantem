import importlib
from os import PathLike
from pathlib import Path

import h5py

from quantem.core.datastructures import Dataset as Dataset
from quantem.core.datastructures import Dataset2d as Dataset2d
from quantem.core.datastructures import Dataset3d as Dataset3d
from quantem.core.datastructures import Dataset4dstem as Dataset4dstem


def read_4dstem(
    file_path: str | PathLike,
    file_type: str | None = None,
    dataset_index: int | None = None,
    **kwargs,
) -> Dataset4dstem:
    """
    File reader for 4D-STEM data

    Parameters
    ----------
    file_path: str | PathLike
        Path to data
    file_type: str
        The type of file reader needed. See rosettasciio for supported formats
        https://hyperspy.org/rosettasciio/supported_formats/index.html
    dataset_index: int, optional
        Index of the dataset to load if file contains multiple datasets.
        If None, automatically selects the first 4D dataset found.
    **kwargs: dict
        Additional keyword arguments to pass to the Dataset4dstem constructor.

    Returns
    --------
    Dataset4dstem
    """
    if file_type is None:
        file_type = Path(file_path).suffix.lower().lstrip(".")

    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader
    data_list = file_reader(file_path)

    # If specific index provided, use it
    if dataset_index is not None:
        imported_data = data_list[dataset_index]
        if imported_data["data"].ndim != 4:
            raise ValueError(
                f"Dataset at index {dataset_index} has {imported_data['data'].ndim} dimensions, "
                f"expected 4D. Shape: {imported_data['data'].shape}"
            )
    else:
        # Automatically find first 4D dataset
        four_d_datasets = [(i, d) for i, d in enumerate(data_list) if d["data"].ndim == 4]

        if len(four_d_datasets) == 0:
            print(f"No 4D datasets found in {file_path}. Available datasets:")
            for i, d in enumerate(data_list):
                print(f"  Dataset {i}: shape {d['data'].shape}, ndim={d['data'].ndim}")
            raise ValueError("No 4D dataset found in file")

        dataset_index, imported_data = four_d_datasets[0]

        if len(data_list) > 1:
            print(
                f"File contains {len(data_list)} dataset(s). Using dataset {dataset_index} with shape {imported_data['data'].shape}"
            )

    imported_axes = imported_data["axes"]

    sampling = kwargs.pop(
        "sampling",
        [ax["scale"] for ax in imported_axes],
    )
    origin = kwargs.pop(
        "origin",
        [ax["offset"] for ax in imported_axes],
    )
    units = kwargs.pop(
        "units",
        ["pixels" if ax["units"] == "1" else ax["units"] for ax in imported_axes],
    )

    dataset = Dataset4dstem.from_array(
        array=imported_data["data"],
        sampling=sampling,
        origin=origin,
        units=units,
        **kwargs,
    )

    return dataset


def read_2d(
    file_path: str | PathLike,
    file_type: str | None = None,
) -> Dataset2d:
    """
    File reader for images

    Parameters
    ----------
    file_path: str | PathLike
        Path to data
    file_type: str
        The type of file reader needed. See rosettasciio for supported formats
        https://hyperspy.org/rosettasciio/supported_formats/index.html

    Returns
    --------
    Dataset2d
    """
    path = Path(file_path)

    if file_type is None:
        file_type = path.suffix.lower().lstrip(".")
    file_type_norm = str(file_type).lower().lstrip(".")

    ext_to_rsciio_module = {
        "dm3": "digitalmicrograph",
        "dm4": "digitalmicrograph",
        "tif": "image",
        "tiff": "image",
        "png": "image",
        "jpg": "image",
        "jpeg": "image",
        "bmp": "image",
        "gif": "image",
    }

    module_name = ext_to_rsciio_module.get(file_type_norm, file_type_norm)

    try:
        module = importlib.import_module(f"rsciio.{module_name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not import RosettaSciIO reader 'rsciio.{module_name}' "
            f"(inferred from file_type='{file_type_norm}')."
        ) from e

    file_reader = getattr(module, "file_reader", None)
    if file_reader is None:
        raise AttributeError(f"RosettaSciIO module 'rsciio.{module_name}' has no 'file_reader'.")

    imported = file_reader(path)
    imported_list = [imported] if isinstance(imported, dict) else list(imported)
    if not imported_list:
        raise ValueError(f"RosettaSciIO reader 'rsciio.{module_name}' returned no datasets for {path}.")

    selected = None
    for item in imported_list:
        data = item.get("data", None) if isinstance(item, dict) else None
        if data is not None and getattr(data, "ndim", None) == 2:
            selected = item
            break
    if selected is None:
        selected = imported_list[0]

    if not isinstance(selected, dict) or "data" not in selected:
        raise ValueError(f"Unexpected RosettaSciIO return structure for {path} using 'rsciio.{module_name}'.")

    axes = selected.get("axes", [])
    if not isinstance(axes, list):
        axes = []

    def _axis_get(i: int, key: str, default: object) -> object:
        if i < len(axes) and isinstance(axes[i], dict):
            return axes[i].get(key, default)
        return default

    def _to_float(x: object, default: float) -> float:
        try:
            return float(x)  # type: ignore[arg-type]
        except Exception:
            return float(default)

    def _to_unit_str(u: object) -> str:
        if isinstance(u, (list, tuple)):
            u = u[0] if len(u) > 0 else ""
        if u is None:
            return "pixels"
        s = str(u).strip()
        return s if s else "pixels"

    sampling = [
        _to_float(_axis_get(0, "scale", 1.0), 1.0),
        _to_float(_axis_get(1, "scale", 1.0), 1.0),
    ]
    origin = [
        _to_float(_axis_get(0, "offset", 0.0), 0.0),
        _to_float(_axis_get(1, "offset", 0.0), 0.0),
    ]
    units = [
        _to_unit_str(_axis_get(0, "units", "pixels")),
        _to_unit_str(_axis_get(1, "units", "pixels")),
    ]

    metadata = selected.get("metadata", {})
    name = path.stem
    signal_units = "arb. units"

    if isinstance(metadata, dict):
        general = metadata.get("General", {})
        if isinstance(general, dict):
            title = general.get("title", None)
            if isinstance(title, str) and title.strip():
                name = title.strip()

        signal = metadata.get("Signal", {})
        if isinstance(signal, dict):
            q = signal.get("quantity", None)
            if isinstance(q, str) and q.strip():
                signal_units = q.strip()

    dataset = Dataset2d.from_array(
        array=selected["data"],
        name=name,
        sampling=sampling,
        origin=origin,
        units=units,
        signal_units=signal_units,
    )
    dataset.file_path = str(path)

    try:
        dataset.metadata.setdefault("rsciio", {})
        dataset.metadata["rsciio"]["module"] = module_name
        dataset.metadata["rsciio"]["file_type"] = file_type_norm
        dataset.metadata["rsciio"]["metadata"] = selected.get("metadata", {})
        dataset.metadata["rsciio"]["original_metadata"] = selected.get("original_metadata", {})
    except Exception:
        pass

    return dataset


def read_emdfile_to_4dstem(
    file_path: str | PathLike,
    data_keys: list[str] | None = None,
    calibration_keys: list[str] | None = None,
) -> Dataset4dstem:
    """
    File reader for legacy `emdFile` / `py4DSTEM` files.

    Parameters
    ----------
    file_path: str | PathLike
        Path to data

    Returns
    --------
    Dataset4dstem
    """
    with h5py.File(file_path, "r") as file:
        # Access the data directly
        data_keys = ["datacube_root", "datacube", "data"] if data_keys is None else data_keys
        print("keys: ", data_keys)
        try:
            data = file
            for key in data_keys:
                data = data[key]  # type: ignore
        except KeyError:
            raise KeyError(f"Could not find key {data_keys} in {file_path}")

        # Access calibration values directly
        calibration_keys = (
            ["datacube_root", "metadatabundle", "calibration"]
            if calibration_keys is None
            else calibration_keys
        )
        try:
            calibration = file
            for key in calibration_keys:
                calibration = calibration[key]  # type: ignore
        except KeyError:
            raise KeyError(f"Could not find calibration key {calibration_keys} in {file_path}")
        r_pixel_size = calibration["R_pixel_size"][()]  # type: ignore
        q_pixel_size = calibration["Q_pixel_size"][()]  # type: ignore
        r_pixel_units = calibration["R_pixel_units"][()]  # type: ignore
        q_pixel_units = calibration["Q_pixel_units"][()]  # type: ignore

        dataset = Dataset4dstem.from_array(
            array=data,
            sampling=[r_pixel_size, r_pixel_size, q_pixel_size, q_pixel_size],
            units=[r_pixel_units, r_pixel_units, q_pixel_units, q_pixel_units],
        )
    dataset.file_path = file_path

    return dataset
