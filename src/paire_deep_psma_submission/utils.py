import json
from itertools import chain
from pathlib import Path
from typing import Any, Sequence, Union

from monai.utils.misc import ensure_tuple


def find_file_path(dir_path: Union[str, Path], ext: Union[str, Sequence[str]] = "*") -> Path:
    exts = ensure_tuple(ext)
    iterator = chain(*[Path(dir_path).glob(f"*{ext}") for ext in exts])
    file_paths = list(iterator)

    if not file_paths:
        raise FileNotFoundError(f"No files found in {dir_path} matching {', '.join(exts)}.")
    if len(file_paths) > 1:
        raise ValueError(f"Multiple files found in {dir_path} matching {', '.join(exts)}: {file_paths}")

    return file_paths[0]


def load_json(file_path: Union[str, Path]) -> Any:
    with open(file_path, "r") as f:
        return json.load(f)
