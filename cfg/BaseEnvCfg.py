from dataclasses import MISSING
import numpy as np

from isaaclab.utils.configclass import configclass
"""Rest everything follows."""
from isaaclab.envs import ManagerBasedRLEnvCfg
from cfg.BaseCfg import CommandsCfg, TerminationsCfg, ActionsCfg

from cfg.BaseTaskCfg import(
    SCENE_CLASSES,
    OBS_CLASSES,
    REW_CLASSES,
    EVENT_CLASSES,
    TARGET_OBJECT_NAME,
)

##
# Environment configuration
##

@configclass
class DynamicEnvCfg(ManagerBasedRLEnvCfg):
    scene: object             = MISSING
    observations: object      = MISSING
    commands: CommandsCfg = CommandsCfg()
    actions: object           = MISSING
    rewards: object           = MISSING
    terminations: TerminationsCfg = TerminationsCfg()
    events: object            = MISSING
    recorders: object         = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 300
        # step_dt = 5 & self.max_episode_length = episode_length_s / step_dt
        self.episode_length_s = 5 * 3 # 5 * skill_length
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 2**24


import ast

def extract_batched_guide_info(seg_maps: np.ndarray,
                               info_list: list, name: str) -> list[list[dict]]:
    """
    Batched segmentation에서 각 env별로 guide 정보를 추출합니다.
    - symbolic guide (class=="guide") → bbox 포함
    - text guide     (class=="guide_text") → bbox 제외

    Args:
        seg_maps: np.ndarray, shape=(N, H, W, 4) 또는 (N, H, W)
        id_to_label: { "(R, G, B, A)": {"class":..., "guide_text":...}, … }

    Returns:
        List[N] of lists of dicts with keys:
          - 'label' (str)
          - 'class' ('guide' or 'guide_text')
          - 'bbox'  ([x0,y0,x1,y1], only for symbolic guide)
    """
    N = seg_maps.shape[0]
    results = [[] for _ in range(N)]
    is_color = (seg_maps.ndim == 4 and seg_maps.shape[-1] == 4)

    for i in range(N):
        id_to_label = info_list[i]['instance_segmentation_fast']['idToSemantics']
        for color_str, info in id_to_label.items():
            if 'guide_text' in info:
                kind  = 'guide_text'
                label = info['guide_text']
                color = color_str
                if is_color:
                    color_arr = np.array(color, dtype=seg_maps.dtype).reshape(1,1,1,4)
                    mask = np.all(seg_maps == color_arr, axis=-1)  # (N,H,W)
                else:
                    mask = (seg_maps == color[0])                  # (N,H,W)

                m = mask[i]
                if not m.any():
                    continue
                ys, xs = np.nonzero(m)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                results[i].append({'class': kind, 'label': label, 'center': [(x0+x1)/2,(y0+y1)/2]})

            # 심볼릭 가이드: info['class']=='guide' 인 경우
            elif 'guide' in info:
                kind  = 'guide'
                label = info.get(kind)
                color = color_str
                # color = tuple(ast.literal_eval(color_str))

                # 픽셀 매칭 마스크 생성
                if is_color:
                    color_arr = np.array(color, dtype=seg_maps.dtype).reshape(1,1,1,4)
                    mask = np.all(seg_maps == color_arr, axis=-1)  # (N,H,W)
                else:
                    mask = (seg_maps == color[0])                  # (N,H,W)

                # 각 env별로 bbox 계산
                m = mask[i]
                if not m.any():
                    continue
                ys, xs = np.nonzero(m)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                results[i].append({
                    'class': kind,
                    'label': label,
                    'bbox': [x0, y0, x1, y1]
                })
        # 그 외 클래스는 무시

    return results
