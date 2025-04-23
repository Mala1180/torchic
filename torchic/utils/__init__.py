import logging

from torch import accelerator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("torchic")

DEVICE = accelerator.current_accelerator().type if accelerator.is_available() else "cpu"
logger.info(f"Using {DEVICE} device")
