from src.algorithms.common.pixel_control import PixelControlAlgorithm
from src.algorithms.sac_bbf.sac_bbf import SACBBFAgent
from src.algorithms.sac_bbf.sac_bbf import SACBBFConfig


class SACBBFAlgorithm(PixelControlAlgorithm):
    agent_cls = SACBBFAgent
    config_cls = SACBBFConfig

__all__ = ["SACBBFAlgorithm"]
