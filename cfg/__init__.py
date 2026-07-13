import gymnasium as gym

gym.register(
    id='SceneCabinet-TopDrawer-interact-v0',
    entry_point='custom_lab.envs.manager_based_rl_step_env:ManagerBasedRLStepEnv',
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scene1Cfg:CabinetTopDrawerSkillEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id='SceneCabinet-BottomDrawer-interact-v0',
    entry_point='custom_lab.envs.manager_based_rl_step_env:ManagerBasedRLStepEnv',
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scene1Cfg:CabinetBottomDrawerSkillEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id='SceneCabinet-LeftDoor-interact-v0',
    entry_point='custom_lab.envs.manager_based_rl_step_env:ManagerBasedRLStepEnv',
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scene1Cfg:CabinetLeftDoorSkillEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id='SceneCabinet-RightDoor-interact-v0',
    entry_point='custom_lab.envs.manager_based_rl_step_env:ManagerBasedRLStepEnv',
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scene1Cfg:CabinetRightDoorSkillEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id='SceneCabinet-TopDrawer-continuous-v0',
    entry_point='custom_lab.envs.manager_based_rl_step_env:ManagerBasedContinuousEnv',
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scene1Cfg:CabinetTopDrawerContinuousEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id='SceneCabinet-BottomDrawer-continuous-v0',
    entry_point='custom_lab.envs.manager_based_rl_step_env:ManagerBasedContinuousEnv',
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scene1Cfg:CabinetBottomDrawerContinuousEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id='SceneCabinet-LeftDoor-continuous-v0',
    entry_point='custom_lab.envs.manager_based_rl_step_env:ManagerBasedContinuousEnv',
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scene1Cfg:CabinetLeftDoorContinuousEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id='SceneCabinet-RightDoor-continuous-v0',
    entry_point='custom_lab.envs.manager_based_rl_step_env:ManagerBasedContinuousEnv',
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scene1Cfg:CabinetRightDoorContinuousEnvCfg",
    },
    disable_env_checker=True,
)   