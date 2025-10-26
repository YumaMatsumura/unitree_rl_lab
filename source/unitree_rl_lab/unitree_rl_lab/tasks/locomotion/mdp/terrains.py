from dataclasses import MISSING
import numpy as np

from isaaclab.utils import configclass
from isaaclab.terrains import SubTerrainBaseCfg


@configclass
class HfBaseCfg(SubTerrainBaseCfg):
    size: tuple = (8.0, 8.0)  # 1タイルの広さ [m]（X: 幅、Y: 奥行き）
    horizontal_scale: float = 0.05  # XY解像度 [m/cell]
    vertical_scale: float = 0.02  # 高さ量子化 [m/段]

# ===== Climbing down =====
@configclass
class ClimbDownTerrainCfg(HfBaseCfg):
    function = hf_climb_down
    box_h: float = 0.5
    edge_offset: float = 2.0  # 台の端までの長さ [m]


@height_field_to_mesh
def hf_climb_down(difficulty: float, cfg: ClimbDownTerrainCfg) -> np.ndarray:
    W = int(cfg.size[0] / cfg.horizontal_scale)
    L = int(cfg.size[1] / cfg.horizontal_scale)
    hf = np.zeros((W, L), dtype=np.int16)
    h = int(cfg.box_h / cfg.vertical_scale)
    edge = int(cfg.edge_offset / cfg.horizontal_scale)
    # 手前に台、奥は地面
    hf[:edge, :] = h
    return hf
