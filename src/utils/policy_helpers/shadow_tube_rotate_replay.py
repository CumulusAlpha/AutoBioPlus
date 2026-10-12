"""ShadowTubeRotateReplay: replay support for the task policy."""

from __future__ import annotations

import numpy as np
import torch

from src.utils.policy_helpers.shadow_tube_rotate_common import DEFAULT_GRASP_PAYLOAD
from src.utils.kinematics import batched_interpolate_linear


class ShadowTubeRotateReplay:
    """Provide replay methods used by the task policy."""

    def _build_trajectory(self) -> torch.Tensor:
        payload = self._load_payload()
        phase_trajs = [
            self._interpolate_sequence(self._to_tensor(payload[key]), self.interp_steps)
            for key in self._PHASE_KEYS
        ]
        chunks = [phase_trajs[0]]
        chunks.append(self._interpolate_pair(phase_trajs[0][-1], phase_trajs[1][0])[1:])
        chunks.append(phase_trajs[1][1:])
        chunks.append(self._interpolate_pair(phase_trajs[1][-1], phase_trajs[2][0])[1:])
        chunks.append(phase_trajs[2][1:])
        return torch.cat(chunks, dim=0).to(device=self.device, dtype=torch.float32)

    def _load_payload(self) -> dict[str, np.ndarray]:
        payload = dict(DEFAULT_GRASP_PAYLOAD)
        for key in self._PHASE_KEYS:
            if key not in payload:
                raise KeyError(f"Missing key in embedded grasp payload: {key}")
            value = np.asarray(payload[key], dtype=np.float32)
            if value.ndim != 2:
                raise ValueError(f"{key} must be 2D, got {value.shape}")
            if value.shape[1] > self.expected_row_dim:
                raise ValueError(
                    f"{key} has {value.shape[1]} action dims, but env action_dim is "
                    f"{self.expected_row_dim}."
                )
            if value.shape[1] < self.expected_row_dim:
                padded = np.zeros(
                    (value.shape[0], self.expected_row_dim), dtype=np.float32
                )
                padded[:, : value.shape[1]] = value
                value = padded
            payload[key] = value
        return payload

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _extract_qpos(self, obs, key: str) -> torch.Tensor:
        qpos = None
        try:
            if key in obs:
                qpos = obs[key]
        except Exception:
            pass
        if qpos is not None and isinstance(qpos, dict):
            qpos = qpos.get(key, qpos)
        if qpos is None:
            raise KeyError(f"Observation missing '{key}'.")
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.as_tensor(qpos, device=self.device, dtype=torch.float32)
        else:
            qpos = qpos.to(device=self.device, dtype=torch.float32)
        if qpos.ndim == 3:
            qpos = qpos[:, 0, :]
        elif qpos.ndim == 1:
            qpos = qpos.unsqueeze(0).repeat(self.num_envs, 1)
        elif qpos.ndim != 2:
            raise ValueError(f"Unexpected {key} shape: {tuple(qpos.shape)}")
        return qpos

    def _interpolate_sequence(self, seq: torch.Tensor, num_steps: int) -> torch.Tensor:
        if seq.shape[0] == 1:
            return seq
        pieces = []
        for i in range(seq.shape[0] - 1):
            pair = self._interpolate_pair(seq[i], seq[i + 1], num_steps=num_steps)
            if i > 0:
                pair = pair[1:]
            pieces.append(pair)
        return torch.cat(pieces, dim=0)

    def _interpolate_pair(
        self,
        start_row: torch.Tensor,
        end_row: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        num_steps = self.interp_steps if num_steps is None else int(num_steps)
        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos=start_row[:3].unsqueeze(0),
            end_pos=end_row[:3].unsqueeze(0),
            start_quat=start_row[3:7].unsqueeze(0),
            end_quat=end_row[3:7].unsqueeze(0),
            num_steps=num_steps,
        )
        t = torch.linspace(
            0.0, 1.0, num_steps + 1, device=self.device, dtype=torch.float32
        )
        tail = (1.0 - t[:, None]) * start_row[7:].unsqueeze(0) + t[:, None] * end_row[
            7:
        ].unsqueeze(0)
        return torch.cat((pos_traj[0], quat_traj[0], tail), dim=-1)
