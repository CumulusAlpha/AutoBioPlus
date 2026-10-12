import abc
from collections.abc import Sequence
from typing import Dict

BATCH_INFERENCE_PROTOCOL_VERSION = 1
BATCH_INFERENCE_METADATA_KEY = "openpi_batch_inference"
BATCH_INFERENCE_REQUEST_KEY = "__openpi_batch_request__"
BATCH_INFERENCE_RESPONSE_KEY = "__openpi_batch_response__"


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def infer_batch(self, observations: Sequence[Dict]) -> list[Dict]:
        """Infer actions for multiple independent observations."""
        return [self.infer(obs) for obs in observations]

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
