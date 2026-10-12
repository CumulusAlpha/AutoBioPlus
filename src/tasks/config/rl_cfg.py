from mjlab.rl import RslRlOnPolicyRunnerCfg


def make_dummy_rl_cfg(**kwargs) -> RslRlOnPolicyRunnerCfg:
    """Create an independent placeholder runner config for a scripted task."""
    return RslRlOnPolicyRunnerCfg(**kwargs)
