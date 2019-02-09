import numpy as np
import pandas as pd
from io import BytesIO, StringIO


def to_bytes(object_) -> bytes:
    if object_ is None:
        return b""
    elif isinstance(object_, pd.DataFrame):
        return df_to_bytes(object_)
    elif isinstance(object_, np.ndarray):
        return np_to_bytes(object_)
    else:
        raise AssertionError(
            f"Cannot save object {object_} to bytes.")


def from_bytes(bytes_: bytes):
    if bytes_ is None:
        return None
    if len(bytes_) == 0:
        return None
    if bytes_[:6] == b"\x93NUMPY":
        return np_from_bytes(bytes_)
    return df_from_bytes(bytes_)


def np_to_bytes(arr) -> bytes:
    f = BytesIO()
    np.save(f, arr, allow_pickle=False)
    f.seek(0)
    arr_bytes = f.read()
    f.close()
    print(arr_bytes)
    return arr_bytes


_primitive_types = [int, float, str]


def np_from_bytes(arr_bytes):
    f = BytesIO()
    f.write(arr_bytes)
    f.seek(0)
    arr = np.load(f)
    f.close()
    for type_ in _primitive_types:
        try:
            typed_arr = type_(arr)
            if np.array_equal(typed_arr, arr):
                return typed_arr
        except (TypeError, ValueError):
            pass
    return arr


def df_to_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode()


def df_from_bytes(bytes_: bytes):
    try:
        s = StringIO(bytes_.decode())
        s.seek(0)
        return pd.read_csv(s)
    except UnicodeDecodeError:
        raise AssertionError("Not a DataFrame.")
