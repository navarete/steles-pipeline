# steles_pipeline/__init__.py

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DB_DIR      = PROJECT_DIR / "db"

from .steles_base    import STELES_BasePipeline
from .steles_part_I  import STELES_Pipeline
from .steles_part_II import STELES_Pipeline_PartII

__all__ = [
    "STELES_Pipeline",
    "STELES_Pipeline_PartII",
]