import logging
import time
from collections.abc import Sequence
from typing import Dict, Optional, Tuple

import websockets.sync.client
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def supports_batch_inference(self) -> bool:
        return (
            self._server_metadata.get(_base_policy.BATCH_INFERENCE_METADATA_KEY)
            == _base_policy.BATCH_INFERENCE_PROTOCOL_VERSION
        )

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        return self._infer_rpc(obs)

    @override
    def infer_batch(self, observations: Sequence[Dict]) -> list[Dict]:  # noqa: UP006
        if not observations:
            raise ValueError("Batch inference requires at least one observation.")
        if not self.supports_batch_inference():
            raise RuntimeError("The OpenPI server does not support batched inference.")
        response = self._infer_rpc({_base_policy.BATCH_INFERENCE_REQUEST_KEY: list(observations)})
        if not isinstance(response, dict):
            raise TypeError(f"Invalid batched inference response type: {type(response)!r}.")
        results = response.get(_base_policy.BATCH_INFERENCE_RESPONSE_KEY)
        if not isinstance(results, list) or len(results) != len(observations):
            raise ValueError(
                "Batched inference response contains "
                f"{len(results) if isinstance(results, list) else 'invalid'} result(s); "
                f"expected {len(observations)}."
            )
        return results

    def _infer_rpc(self, request: Dict) -> Dict:
        data = self._packer.pack(request)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
