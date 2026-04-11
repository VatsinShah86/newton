import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third-party" / "oculus_reader"))

from oculus_reader.reader import OculusReader
