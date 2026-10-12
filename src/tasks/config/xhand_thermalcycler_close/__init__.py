from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import make_xhand_thermalcycler_env_cfg
from ..rl_cfg import make_dummy_rl_cfg

register_mjlab_task(
    task_id="XHand-ThermalCycler-Close",
    env_cfg=make_xhand_thermalcycler_env_cfg(mode="close"),
    play_env_cfg=make_xhand_thermalcycler_env_cfg(play=True, mode="close"),
    rl_cfg=make_dummy_rl_cfg(num_steps_per_env=1, max_iterations=1),
)
