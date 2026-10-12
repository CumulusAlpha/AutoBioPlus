from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import make_xhand_thermalcycler_open_env_cfg
from ..rl_cfg import make_dummy_rl_cfg


register_mjlab_task(
    task_id="XHand-ThermalCycler-Open",
    env_cfg=make_xhand_thermalcycler_open_env_cfg(),
    play_env_cfg=make_xhand_thermalcycler_open_env_cfg(play=True),
    rl_cfg=make_dummy_rl_cfg(experiment_name='dummy'),
)
