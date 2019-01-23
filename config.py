from os.path import realpath, dirname
from pathlib import Path

REPO_NAME = "config"
REPO_PATH = Path(dirname(realpath(__file__)))
DATA_PATH = REPO_PATH / "data"
if Path(DATA_PATH / REPO_NAME).exists():
    DATA_PATH = DATA_PATH / REPO_NAME / "data"

OUTPUT_PATH = DATA_PATH / "output"


