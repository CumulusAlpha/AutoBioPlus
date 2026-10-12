"""Task goal display, policy reset hooks and the shared Viser viewer."""

from __future__ import annotations

import html

import numpy as np
import torch
import viser
from mjlab.viewer import ViserPlayViewer

from src.tasks.task_goals import (
    TASK_PROMPT_TEMPLATES,
    extract_task_goal_values,
    task_goal_names,
    task_prompts_for_envs,
)


def reset_policy_after_env_done(env, policy, on_env_reset=None):
    reset_envs_fn = getattr(policy, "reset_envs", None)
    reset_fn = getattr(policy, "reset", None)
    if reset_envs_fn is None and reset_fn is None and on_env_reset is None:
        return

    original_step = env.step

    def step_with_policy_reset(action):
        out = original_step(action)
        if len(out) == 4:
            done = out[2]
        else:
            done = out[2] | out[3]
        done = torch.as_tensor(done, device=env.unwrapped.device, dtype=torch.bool)
        if done.any():
            env_ids = torch.where(done)[0]
            if reset_envs_fn is not None:
                reset_envs_fn(env_ids)
            elif bool(done.all().item()):
                reset_fn()
            if on_env_reset is not None:
                on_env_reset(env_ids)
        return out

    env.step = step_with_policy_reset


class PlayTaskGoalDisplay:
    """Expose reset-selected task goals and prompts in play viewers."""

    def __init__(
        self,
        task_id: str,
        env,
        server: viser.ViserServer | None,
        *,
        prompt_override: str | None = None,
    ) -> None:
        self.task_id = task_id
        self.env = env
        self.prompt_override = prompt_override
        self.num_envs = int(env.unwrapped.num_envs)
        self.goal_names = task_goal_names(task_id) or ()
        self.goal_records: list[dict[str, float | int] | None] = [None] * self.num_envs
        self.prompts = [""] * self.num_envs
        self.goal_info = None
        self._viewer_scene = None
        self._displayed_env_id: int | None = None

        if server is not None:
            with server.gui.add_folder("Task Goal"):
                self.goal_info = server.gui.add_html("<div></div>")

        self._install_manual_reset_hook()
        self.refresh_envs()

    def _install_manual_reset_hook(self) -> None:
        original_reset = self.env.reset

        def reset_with_goal_refresh(*args, **kwargs):
            out = original_reset(*args, **kwargs)
            self.refresh_envs()
            return out

        self.env.reset = reset_with_goal_refresh

    def _selected_env_id(self) -> int:
        if self._viewer_scene is None:
            return 0
        return int(self._viewer_scene.env_idx)

    def bind_viewer_scene(self, viewer_scene) -> None:
        """Use the Viser Scene selector as the task-goal environment selector."""
        self._viewer_scene = viewer_scene
        self.sync_from_viewer_scene()

    def sync_from_viewer_scene(self) -> None:
        if self._viewer_scene is None:
            return
        env_id = int(self._viewer_scene.env_idx)
        if env_id == self._displayed_env_id:
            return
        self._update_selected_env()

    def _update_selected_env(self) -> None:
        if self.goal_info is None:
            return
        env_id = self._selected_env_id()
        self._displayed_env_id = env_id
        record = self.goal_records[env_id]
        parameters = (
            ", ".join(f"<code>{html.escape(name)}</code>" for name in self.goal_names)
            or "fixed (none)"
        )
        values = (
            "fixed (none)"
            if record is None
            else ", ".join(
                f"<code>{html.escape(name)}={html.escape(str(value))}</code>"
                for name, value in record.items()
            )
        )
        self.goal_info.content = (
            '<div style="font-size: 0.8rem; line-height: 1.35; overflow-wrap: anywhere; '
            'padding: 0.3rem 0.6rem 0.45rem">'
            f"<div><strong>Task:</strong> <code>{html.escape(self.task_id)}</code></div>"
            f"<div><strong>Parameters:</strong> {parameters}</div>"
            "<div><strong>Prompt template:</strong> "
            f"{html.escape(TASK_PROMPT_TEMPLATES[self.task_id])}</div>"
            f"<div><strong>Environment:</strong> <code>{env_id}</code></div>"
            f"<div><strong>Current values:</strong> {values}</div>"
            f"<div><strong>Prompt:</strong> {html.escape(self.prompts[env_id])}</div>"
            "</div>"
        )

    def refresh_envs(self, env_ids: torch.Tensor | None = None) -> None:
        ids = (
            list(range(self.num_envs))
            if env_ids is None
            else [int(value) for value in env_ids.detach().cpu().reshape(-1).tolist()]
        )
        goals = extract_task_goal_values(self.env.unwrapped, self.task_id)
        prompts = task_prompts_for_envs(
            self.env.unwrapped,
            self.task_id,
            ids,
            override=self.prompt_override,
        )
        goals_cpu = None if goals is None else goals.detach().cpu()

        for env_id, prompt in zip(ids, prompts, strict=True):
            if goals_cpu is None:
                record = None
            else:
                record = {
                    name: value.item()
                    for name, value in zip(
                        self.goal_names,
                        goals_cpu[env_id],
                        strict=True,
                    )
                }
            self.goal_records[env_id] = record
            self.prompts[env_id] = prompt
            print(
                f"[task_goal] env={env_id} parameters="
                f"{record if record is not None else 'fixed (none)'} prompt={prompt!r}"
            )

        if self._selected_env_id() in ids:
            self._update_selected_env()


class DexBioLabViserPlayViewer(ViserPlayViewer):
    """Viser viewer with one stationary-scene environment selector."""

    def __init__(
        self,
        env,
        policy,
        *,
        goal_display,
        viser_server,
        frame_rate: float = 60.0,
    ) -> None:
        super().__init__(
            env,
            policy,
            frame_rate=frame_rate,
            viser_server=viser_server,
        )
        self._goal_display = goal_display

    def setup(self) -> None:
        super().setup()
        # Viser's camera tracking recenters by translating the entire rendered
        # scene around a body from the selected environment. Environment
        # selection should instead leave the scene fixed; camera overlays still
        # use scene.env_idx and therefore move their frustums to the selected env.
        self._scene._tracked_body_id = None
        self._scene._scene_offset = np.zeros(3)
        self._scene._request_update()
        self._goal_display.bind_viewer_scene(self._scene)

    def sync_env_to_viewer(self) -> None:
        self._goal_display.sync_from_viewer_scene()
        super().sync_env_to_viewer()

    def close(self) -> None:
        # This viewer is constructed with an external server so the task-goal
        # controls can share it. Stop that server before waiting for queued
        # scene updates; otherwise an interrupted remote session can leave the
        # viewer blocked during thread-pool shutdown.
        if self._term_overlays:
            self._term_overlays.cleanup()
        if self._camera_overlays:
            self._camera_overlays.cleanup()
        if self._external_server:
            self._server.stop()
        self._threadpool.shutdown(wait=True)
