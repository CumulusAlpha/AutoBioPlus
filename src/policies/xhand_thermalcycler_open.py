from .xhand_thermalcycler_close import XHandThermalCyclerClosePolicy


class XHandThermalCyclerOpenPolicy(XHandThermalCyclerClosePolicy):
    pass


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    del replay_path
    return XHandThermalCyclerOpenPolicy(
        action_shape=action_shape,
        device=device,
        env=env,
        loop=False,
        server=server,
    )
