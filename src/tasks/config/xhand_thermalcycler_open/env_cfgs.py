from src.tasks.config.xhand_thermalcycler_close.env_cfgs import (
    make_xhand_thermalcycler_env_cfg,
)


def make_xhand_thermalcycler_open_env_cfg(play: bool = False):
    return make_xhand_thermalcycler_env_cfg(play=play, mode="open")
