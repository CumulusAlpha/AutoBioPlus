from __future__ import annotations

from importlib import import_module
from typing import Any


def policy_module_name_from_task_id(task_id: str) -> str:
    return task_id.replace("-", "_").lower()


def make_scene_policy(
    task_id: str,
    *,
    action_shape: tuple[int, ...],
    device: str,
    env: Any,
    server: Any = None,
    replay_path: str | None = None,
):
    module_name = policy_module_name_from_task_id(task_id)
    import_name = f"{__name__}.{module_name}"
    try:
        module = import_module(import_name)
    except ModuleNotFoundError as exc:
        if exc.name == import_name:
            raise RuntimeError(
                f"No scripted policy module found for task_id={task_id!r}: {import_name}"
            ) from exc
        raise

    make_policy = getattr(module, "make_policy", None)
    if make_policy is None:
        raise RuntimeError(
            f"Policy module {import_name!r} must define make_policy(...)."
        )

    return make_policy(
        action_shape=action_shape,
        device=device,
        env=env,
        server=server,
        replay_path=replay_path,
    )
