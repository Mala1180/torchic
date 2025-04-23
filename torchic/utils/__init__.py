import logging

from torch import cuda, backends

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("torchic")


DEVICE = (
    "cuda" if cuda.is_available() else "mps" if backends.mps.is_available() else "cpu"
)
logger.info(f"Using {DEVICE} device")
