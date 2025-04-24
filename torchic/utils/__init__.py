import logging
from typing import Optional

import torch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("torchic")

current_accelerator: Optional[torch.device] = torch.accelerator.current_accelerator()

DEVICE: str = current_accelerator.type if current_accelerator is not None else "cpu"
logger.info(f"Using {DEVICE} device")
