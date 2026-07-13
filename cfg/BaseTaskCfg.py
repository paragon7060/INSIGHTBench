

###
# For Scene Setting
###
"""Rest everything follows."""
from isaaclab.envs import ManagerBasedRLEnvCfg
from cfg.BaseCfg import CommandsCfg, TerminationsCfg, ActionsCfg, ObsCfg
from cfg.scene1Cfg import Scene1aCfg, Scene1bCfg, Scene1cCfg, Scene1dCfg, Scene1eCfg
from cfg.scene1Cfg import Reward1aCfg, Reward1bCfg, Reward1cCfg, Reward1dCfg, Reward1eCfg
from cfg.scene1Cfg import EventCabinetCfg, Event1aCfg, Event1bCfg, Event1cCfg, Event1dCfg, Event1eCfg, EventCabinetColorCfg, EventCabinetTestCfg, EventCabinetNoposCfg, EventCabinetNoposColorCfg
from cfg.scene3Cfg import EventDoorCfg, EventDoorColorCfg, EventDoorTestCfg, EventDoorNoposCfg, EventDoorNoposColorCfg
from cfg.scene5Cfg import Scene5aCfg, Scene5bCfg, Scene5cCfg, Scene5dCfg, Scene5eCfg
from cfg.Scene5ExtCfg import SceneBottleExtCfg
from cfg.scene5Cfg import Reward5Cfg, Reward5CloseCfg, Event5Cfg, Event5RobotFricHighCfg, EventBottleColorCfg, EventBottleTestCfg, EventBottleNoposCfg, EventBottleNoposColorCfg
# from cfg.scene2Cfg

SCENE_CLASSES = {
    "1a": Scene1aCfg,
    "1b": Scene1bCfg,
    "1c": Scene1cCfg,
    "1d": Scene1dCfg,
    "1e": Scene1eCfg,
    # "2a": Scene2aCfg,
    # "2b": Scene2bCfg,
    "5a": Scene5aCfg,
    "5b": Scene5bCfg,
    "5c": Scene5cCfg,
    "5d": Scene5dCfg,
    "5e": Scene5eCfg,
    "5ext": SceneBottleExtCfg,
}

OBS_CLASSES = {
    "1": ObsCfg,
    # "2": Scene2ObsCfg,
    # "3": Scene3ObsCfg,
    # "4": Scene4ObsCfg,
    # "5": Scene5ObsCfg,
}

COM_CLASSES = {
    "1": CommandsCfg,
}

REW_CLASSES = {
    "1a": Reward1aCfg,
    "1b": Reward1bCfg,
    "1c": Reward1cCfg,
    "1d": Reward1dCfg,
    "1e": Reward1eCfg,
    # "2a": Reward2aCfg,
    "5a": Reward5Cfg,
    "5b": Reward5Cfg,
    "5c": Reward5Cfg,
    "5d": Reward5CloseCfg,
    "5e": Reward5CloseCfg,
    "5ext": Reward5Cfg,
}

EVENT_CLASSES = {
    "1a": Event1aCfg,
    "1b": Event1bCfg,
    "1c": Event1cCfg,
    "1d": Event1dCfg,
    "1e": Event1eCfg,
    "1":EventCabinetCfg,
    "1ext":EventCabinetCfg,
    "3":EventDoorCfg,
    "3ext":EventDoorCfg,
    "5a": Event5Cfg,
    "5b": Event5Cfg,
    "5c": Event5Cfg,
    "5d": Event5Cfg,
    "5e": Event5Cfg,
    "5ext": Event5RobotFricHighCfg,
    "5": Event5RobotFricHighCfg,
    "1color":EventCabinetColorCfg,
    "3color":EventDoorColorCfg,
    "5color":EventBottleColorCfg,
    "1test":EventCabinetTestCfg,
    "3test":EventDoorTestCfg,
    "5test":EventBottleTestCfg,
    "1nopos":EventCabinetNoposCfg,
    "3nopos":EventDoorNoposCfg,
    "5nopos":EventBottleNoposCfg,
    "1nopos_color":EventCabinetNoposColorCfg,
    "3nopos_color":EventDoorNoposColorCfg,
    "5nopos_color":EventBottleNoposColorCfg,
}

TARGET_OBJECT_NAME={
    "1":"cabinet",
    "2":"microwave",
    "3":"door",
    "4":"washingmachine",
    "5":"bottle",
}

REWARD_FOR_SCENE_KEY = {
    "1": 0.5,
    # "2": 0.003,
    "3": 0.1,
    # "4": 0.003,
    "5": 0.003,
}

SCENE_TASK_PROMPT = {
    "1ext": "Open the cabinet.",
    "3a": "Open the door.",
    "3b": "Open the door.",
    "3c": "Open the door.",
    "3d": "Open the door.",
    "5a": "Open the bottle.",
    "5b": "Open the bottle.",
    "5c": "Open the bottle.",
    "5d": "Close the bottle.",
    "5e": "Close the bottle.",
    "5f": "Close the bottle.",
    "5g": "Open the bottle.",
    "5h": "Close the bottle cap.",
}

CABINET_TEST_USD = ['45677','46130'] # '19179','45135','31249','34178','45194',
DOOR_TEST_USD = ['99689669960001','99689669962036','99692809960003'] #'99660089960018','99660089962003','99660099962042',
BOTTLE_TEST_USD = ['b4','b17','3b','8b','14b','b3','b8'] ##'b4','b17','3b','8b','14b','b3','b8'


SCENE_TASK_PROMPT_GUIDE = {
    "1ext": "Open the cabinet following the guide.",
    "3a": "Open the door following the guide.",
    "3b": "Open the door following the guide.",
    "3c": "Open the door following the guide.",
    "3d": "Open the door following the guide.",
    "5a": "Open the bottle following the guide.",
    "5b": "Open the bottle following the guide.",
    "5c": "Open the bottle following the guide.",
    "5d": "Close the bottle following the guide.",
    "5e": "Close the bottle following the guide.",
    "5f": "Close the bottle following the guide.",
    "5g": "Open the bottle following the guide.",
    "5h": "Close the bottle cap following the guide.",
}


SCENE_TASK_PROMPT_INSTRUCTION = {
    "1ext": "Find the arrow guide and open the indicated drawer",
    "3a": "Open the door, rotate clockwise and push.",
    "3b": "Open the door, rotate counter-clockwise and push.",
    "3c": "Open the door, rotate clockwise and pull.",
    "3d": "Open the door, rotate counter-clockwise and pull.",
    "5a": "Grip the cap on the sides indicated by the 'squeeze' arrow and open the bottle in counter-clockwise direction.",
    "5b": "Open the bottle in counter-clockwise direction.",
    "5c": "Open the bottle in clockwise direction.",
    "5d": "Close the bottle in clockwise direction.",
    "5e": "Close the bottle in counter-clockwise direction.",
    "5f": "Close the bottle in clockwise direction.",
    "5g": "Grip the cap on the sides indicated by the 'squeeze' arrow and open the bottle in clockwise direction.",
    "5h": "Close the bottle in counter-clockwise direction.",
}


SCENE_TASK_PROMPT_SEM = {
    "1ext": "Find the arrow guide and open the indicated drawer",
    "3a": "Open the door following the text or symbolic guide.",
    "3b": "Open the door following the text or symbolic guide.",
    "3c": "Open the door following the text or symbolic guide.",
    "3d": "Open the door following the text or symbolic guide.",
    "5a": "Open the bottle following the text or symbolic guide.",
    "5b": "Open the bottle following the text or symbolic guide.",
    "5c": "Open the bottle following the text or symbolic guide.",
    "5d": "Close the bottle following the text or symbolic guide.",
    "5e": "Close the bottle following the text or symbolic guide.",
    "5f": "Close the bottle following the text or symbolic guide.",
    "5g": "Open the bottle following the text or symbolic guide.",
    "5h": "Close the bottle cap following the text or symbolic guide.",
}

SCENE_TASK_PROMPT_INSTRUCTION_REVERSE = {
    "1ext": "Find the arrow guide and open the indicated drawer",
    "3a": "Open the door, rotate counter-clockwise and pull.",
    "3b": "Open the door, rotate clockwise and pull.",
    "3c": "Open the door, rotate counter-clockwise and push.",
    "3d": "Open the door, rotate clockwise and push.",
    "5a": "Grip the cap on the sides indicated by the 'squeeze' arrow and open the bottle in clockwise direction.",
    "5b": "Open the bottle in clockwise direction.",
    "5c": "Open the bottle in counter-clockwise direction.",
    "5d": "Close the bottle in counter-clockwise direction.",
    "5e": "Close the bottle in clockwise direction.",
    "5f": "Close the bottle in counter-clockwise direction.",
    "5g": "Grip the cap on the sides indicated by the 'squeeze' arrow and open the bottle in counter-clockwise direction.",
    "5h": "Close the bottle in clockwise direction.",
}