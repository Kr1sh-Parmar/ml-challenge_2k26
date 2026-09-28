"""Paths shared by every stage (v2, v3, er4). Nothing outside this package is assumed except the dataset.

ER_DATA  folder holding train/ and test/ (default: auto-found `student_resource/dataset` above this package)
ER_WORK  caches, models, stage outputs - tens of GB (default: <package>/work)
"""
import os
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]                   # code/business_entity_resolution
SRC = PKG / "src"
WORK = Path(os.environ.get("ER_WORK", PKG / "work"))


def _find_data():
    if os.environ.get("ER_DATA"):
        return Path(os.environ["ER_DATA"])
    for p in [PKG, *PKG.parents]:
        for d in (p / "student_resource" / "dataset", p / "dataset"):
            if (d / "train").is_dir() and (d / "test").is_dir():
                return d
    raise FileNotFoundError("dataset not found: set ER_DATA to the folder holding train/ and test/")


DATA = _find_data()
VALIDATOR = DATA.parent / "utils" / "validate_submission.py"   # the organisers' checker (student_resource/utils)
