import pickle
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from termcolor import cprint

from src import DEXBIOLAB_SRC_ROOT


class RecorderStorage:
    def __init__(self):
        self.buffer_dict = {}

        # logs directory under the parent of DEXBIOLAB_SRC_ROOT
        self.log_dir = DEXBIOLAB_SRC_ROOT.parent / "logs"

    def clear_buffer(self):
        self.buffer_dict = {}

    def _to_numpy_safe(self, x):
        """Safely convert data to NumPy (supports CUDA tensors)."""
        if isinstance(x, torch.Tensor):
            # Move GPU tensors to the CPU first
            if x.device.type != "cpu":
                x = x.detach().cpu()
            else:
                x = x.detach()
            return x.numpy()

        return x

    def append_buffer(self, data_dict):
        for k, v in data_dict.items():
            if k not in self.buffer_dict:
                self.buffer_dict[k] = []

            v = self._to_numpy_safe(v)

            self.buffer_dict[k].append(deepcopy(v))

    def save_recordings(self, other_payload=None):
        if len(self.buffer_dict) == 0:
            cprint("No recordings to save.", "yellow")
            return

        payload = {}

        for k, v in self.buffer_dict.items():
            try:
                payload[k] = np.stack(v, axis=0)
            except Exception:
                # Fall back to an array if stacking fails
                payload[k] = np.array(v)

        if other_payload:
            payload.update(other_payload)

        # Create the logs directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.log_dir / f"recording_{timestamp}.pkl"

        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

        cprint(f"Recordings saved to {save_path}, Clearing buffer.", "yellow")

        self.clear_buffer()
