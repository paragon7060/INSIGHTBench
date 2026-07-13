from typing import Any, Dict

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register(task_key: str, spec: Dict[str, Any]) -> None:
    if task_key in TASK_REGISTRY:
        raise KeyError(f"Duplicate task key: {task_key}")
    TASK_REGISTRY[task_key] = spec


# Base tasks
register(
    "door.open",
    {
        "env_class": "custom_lab.envs.manager_based_rl_step_env:ManagerBasedRLStepEnv",
        "dynamic_env_cfg": False,
        "env_cfg_class": "scene_interact.interact_scene3:CabinetEnvCfg",
        "defaults": {
            "render_mode": None,
            "scene_key": "3",  # optional routing key used in env
        },
    },
)

register(
    "cabinet.pull",
    {
        "env_class": "custom_lab.envs.manager_based_rl_step_env:ManagerBasedRLStepEnv",
        "env_cfg_class": "scene_interact.interact_scene1:CabinetEnvCfg",
        "defaults": {
            "render_mode": None,
            "scene_key": "1",
        },
    },
)

# Extension (USD) variants — constructed via DynamicEnvCfg from BaseEnvCfg and Ext scene builders
register(
    "door.open.ext",
    {
        "env_class": "custom_lab.envs.manager_based_rl_step_env:ManagerBasedRLStepEnv",
        "dynamic_env_cfg": True,
        "defaults": {
            "scene_builder": "cfg.scene3ExtCfg:make_door_scene_cfg",
            "event_cfg": "cfg.scene3Cfg:EventDoorCfg",
            "actions_cfg": "cfg.BaseCfg:ActionsCfg",
            "terminations_cfg": "cfg.BaseCfg:TerminationsCfg",
            "observations_cfg": "cfg.scene3Cfg:Scene3ObsCfg",
            "render_mode": None,
            "scene_key": "3ext",
            # required at runtime: usd_path, guide_path; optional scales/poses
        },
    },
)


def get_task(task_key: str) -> Dict[str, Any]:
    return TASK_REGISTRY[task_key] 