from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import make_wuji_tweezers_manipulate_env_cfg
from ..rl_cfg import make_dummy_rl_cfg

register_mjlab_task(
  task_id="Wuji-Tweezers-Manipulate",
  env_cfg=make_wuji_tweezers_manipulate_env_cfg(),
  play_env_cfg=make_wuji_tweezers_manipulate_env_cfg(play=True),
  rl_cfg=make_dummy_rl_cfg(num_steps_per_env=1, max_iterations=1),
)
