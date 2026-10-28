"""Shared test fixtures: project root on sys.path + loader for numeric-prefix scripts."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_script(relpath: str):
    """Import a module from a file whose name is not a valid identifier (e.g. 04_chunk_text.py)."""
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
