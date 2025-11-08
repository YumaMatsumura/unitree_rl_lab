from dataclasses import MISSING
import numpy as np
from typing import Tuple

from isaaclab.utils import configclass
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh


@configclass
class HfBaseCfg(SubTerrainBaseCfg):
    horizontal_scale: float = 0.1  # XY解像度 [m/cell]
    vertical_scale: float = 0.005  # 高さ量子化 [m/段]

# ===== Climbing down =====
@configclass
class ClimbDownTerrainCfg(HfBaseCfg):
    function = None
    box_height_range: tuple[float, float] = MISSING
    edge_offset: float = MISSING  # 台の端までの長さ [m]


@height_field_to_mesh
def hf_climb_down(difficulty: float, cfg: ClimbDownTerrainCfg) -> np.ndarray:
    min_height, max_height = cfg.box_height_range
    box_height = min_height + difficulty * (max_height - min_height)

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_steps = int(round(box_height / cfg.vertical_scale))
    edge = int(round(cfg.edge_offset / cfg.horizontal_scale))
    edge = int(np.clip(edge, 0, width_pixels))

    # 手前に台、奥は地面
    height_field = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    height_field[:edge, :] = np.int16(height_steps)
    return height_field

ClimbDownTerrainCfg.function = hf_climb_down
