import os
import random
import glob

from pathlib import Path

# Isaac Lab 프로젝트 루트 경로를 환경 변수에서 가져옵니다.
ISAACLAB_ROOT_DIR = os.environ.get("ISAACLAB_PATH", ".") # 환경 변수가 없으면 현재 폴더를 기준으로 함


def cabinet_get_joint_id_from_json(json_info, handle_id):
    # 1. handle → link
    link_id = None
    for k, v in json_info['relation'].items():
        if handle_id in v:
            link_id = k
            break

    if link_id is None:
        raise ValueError(f"No link found for handle_id: {handle_id}")

    # 2. link → joint (find joint where this link is the child)
    joint_id = None
    for j_id, j_info in json_info['joint_body_connections'].items():
        if j_info['child'] == link_id:
            joint_id = j_id
            break

    if joint_id is None:
        raise ValueError(f"No joint found for link_id (child): {link_id}")

    # 3. You can now access the joint type, parent, etc.
    joint_info = json_info['joint_body_connections'][joint_id]
    joint_type = joint_info['joint_type']
    parent_link = joint_info['parents']
    return joint_id

def _cabinet_handle_id_from_asset_path(asset_path: str) -> str | None:
    for part in os.path.basename(asset_path).split("-"):
        if part.startswith("handle_"):
            return part
    return None

def _cabinet_handle_id_from_scene_key(scene_key: str, handles: list[str]) -> str | None:
    if scene_key.isdigit():
        scene_idx = int(scene_key)
    elif len(scene_key) == 2 and scene_key[0] == "1" and scene_key[1].isalpha():
        scene_idx = ord(scene_key[1]) - ord("a")
    else:
        return None

    if scene_idx < 0 or scene_idx >= len(handles):
        raise IndexError(
            f"Cabinet scene_key '{scene_key}' maps to handle index {scene_idx}, "
            f"but only {len(handles)} handle(s) exist: {handles}"
        )
    return handles[scene_idx]

def resolve_cabinet_collect_handle_id(asset_path: str, scene_key: str, json_info: dict, json_path: str) -> str:
    """Resolve the cabinet handle id for collection without treating '1ext' as an index."""
    handles = json_info.get("handle", [])
    if not handles:
        raise ValueError(f"No cabinet handles listed in {json_path}")

    hand_id = _cabinet_handle_id_from_asset_path(asset_path)
    if hand_id is None:
        hand_id = _cabinet_handle_id_from_scene_key(scene_key, handles)
    if hand_id is None and len(handles) == 1:
        hand_id = handles[0]
    if hand_id is None:
        raise ValueError(
            f"Cannot resolve cabinet handle for asset_path='{asset_path}', scene_key='{scene_key}'. "
            "Use a PartManip-style asset id containing 'handle_<id>' or a numeric/1a-style scene key."
        )
    if hand_id not in handles:
        raise ValueError(f"Resolved handle '{hand_id}' is not in {json_path}: {handles}")
    return hand_id

def cabinet_get_link_id_from_json(json_info, handle_id):
    for key, value in json_info['relation'].items():
        if handle_id in value:
            return key
    raise ValueError(f"No link found for handle_id: {handle_id}")

def sample_usd_dir_from_dir(usd_root: str, reversed: bool=False, any: bool=False, object: str=None) -> str:
    if object=="cabinet":
        dictionary = set()
        subdirs = []
        for name in os.listdir(usd_root):
            key = name.split("-")[1]
            if os.path.isdir(os.path.join(usd_root, name)) and key not in dictionary:
                dictionary.add(key)
                subdirs.append(name)
    else:
        subdirs = [subdir for subdir in os.listdir(usd_root) if os.path.isdir(os.path.join(usd_root, subdir))]

    if any:
        # 서브폴더 내 아무 USD라도 있으면 후보에 추가
        candidates = []
        for sub in subdirs:
            usd_files = glob.glob(os.path.join(usd_root, sub, "*.usd"))
            if usd_files:  # 리스트가 비어있지 않으면
                candidates.append(os.path.join(usd_root, sub))
    else:
        usd_name = "mobility_reversed.usd" if reversed else "mobility.usd"
        candidates = []
        for sub in subdirs:
            usd_path = os.path.join(usd_root, sub, usd_name)
            if os.path.isfile(usd_path):
                candidates.append(os.path.join(usd_root, sub))

    if not candidates:
        raise FileNotFoundError(f"No matching USD found under any subfolder of {usd_root}")

    sysrand = random.SystemRandom()
    usd_dir = sysrand.choice(candidates)
    abs_usd_dir = Path(usd_dir).resolve()
    return str(abs_usd_dir)

import json
import numpy as np
from typing import Iterable, Dict, Any


def load_bbox_and_compute_center_and_zmax(json_path: str, part_names: Iterable[str]) -> dict[str, dict]:
    """
    JSON을 한 번 읽고, 각 part에 대해:
      - raw min, max
      - center = (min + max) / 2
      - max_z = max[2]
    """
    with open(json_path, 'r') as f:
        bb = json.load(f)

    result = {}
    for part in part_names:
        min_key = f"{part}_min"
        max_key = f"{part}_max"
        if min_key not in bb or max_key not in bb:
            raise KeyError(f"Expected keys '{min_key}' and '{max_key}' in JSON for part '{part}'")

        mn = np.array(bb[min_key], dtype=float)
        mx = np.array(bb[max_key], dtype=float)

        center = (mn + mx) / 2.0
        max_z = float(mx[2])

        result[part] = {
            "min": mn,
            "max": mx,
            "center": center,
            "max_z": max_z,
        }

    return result

def load_urdf_and_get_joint2_x(urdf_path:str):
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    x_value: float=0.0
    for joint in root.findall("joint"):
        if joint.attrib.get("name") == "joint_2":
            origin_elem = joint.find("origin")
            if origin_elem is not None:
                xyz_str = origin_elem.attrib.get("xyz", "")
                # "0.8133098319472123 -0.02 0.016478278052564438" → [0.8133..., -0.02, 0.0164...]
                xyz_vals = list(map(float, xyz_str.split()))
                x_value = xyz_vals[0]  # x값만 추출
            break
    return x_value

def sample_urdf_get_info(dir_path: str, object: str) -> Dict[str, Any]:
    sysrand = random.SystemRandom()
    #TODO: guide init pos에 (x,y,z)로 뱉어지게 추가해서 'guide_init_quat'를 result 에 추가
    result : Dict[str, Any] = {
        "usd_path": None, "guide_path": None, "json_path": None,
        "asset_init_pos": None, "guide_init_pos": None, "guide_init_quat": None,
        "scale": None, "guide_scale": None,
        "pull_mode": None, "ccw_mode": None, "close_mode": None,
        "scene_key": None, "cover_points": None,
        "joint2_x_position": None
    }

    if object == "bottle":
        ###
        # Guide mode selection
        ##
        """
        Guide mode
            - 0: open text + curved arrow
            - 1: open text + curved arrow reversed
            - 2: open/close text + 2 curved arrows
            - 3: open/close text + 2 curved arrows with reversed orientation
            - 4: squeeze sign + 2 arrows + open arrow (guide_0)
            - 5: squeeze sign + 2 arrows + reversed open arrow (guide_1)
        """
        guide_mode = sysrand.randint(0, 5)
        guide_base_path = f"{ISAACLAB_ROOT_DIR}/Assets/guides/group_guide/guide_mode_"
        guide_path = guide_base_path + str(guide_mode) + ".usd"

        reverse_mode = (guide_mode % 2) == 1
        usd_dir_path = sample_usd_dir_from_dir(dir_path, reversed=reverse_mode,object=object)
        if reverse_mode:
            usd_path = os.path.join(usd_dir_path, "mobility_reversed.usd")
        else:
            usd_path = os.path.join(usd_dir_path, "mobility.usd")
        bbox_path = os.path.join(usd_dir_path, "bounding_box.json")

        cover_points = load_bbox_and_compute_center_and_zmax(bbox_path, ["cover"])
        cx = cover_points["cover"]['center'][0]
        cy = cover_points["cover"]['center'][1]
        cz = cover_points["cover"]['center'][2] + 0.001

        asset_init_pos = (0.5, 0, 0.492)
        guide_init_pos = (cx, cy, cover_points["cover"]["max"][2])

        # compute guide scale from bbox
        xy_scale = 8.5 * (cover_points['cover']['max'] - cover_points['cover']['min'])
        xy_scale[-1] = 0.8
        guide_scale = tuple(xy_scale)

        # close_mode : True => init state open and change reward...?
        close_mode = sysrand.choice([True, False])

        if guide_mode in {0,2} and not close_mode:
            scene_key = "5b"
        elif guide_mode in {0,2} and close_mode:
            scene_key = "5d"
        elif guide_mode in {1,3} and not close_mode:
            scene_key = "5c"
        elif guide_mode in {1,3} and close_mode:
            scene_key = "5e"
        elif guide_mode == 5 and not close_mode:
            scene_key = "5g"
        elif guide_mode == 5 and close_mode:
            scene_key = "5h"
        elif guide_mode == 4 and not close_mode:
            scene_key = "5a"
        elif guide_mode == 4 and close_mode:
            scene_key = "5f"
    
        result["usd_path"] = usd_path
        result["guide_path"] = guide_path
        result["asset_init_pos"] = asset_init_pos
        result["guide_init_pos"] = guide_init_pos
        result["guide_scale"] = guide_scale
        result["close_mode"] = close_mode
        result["scene_key"] = scene_key
        result["cover_points"] = cover_points

    if object == "cabinet":
        usd_dir_path = sample_usd_dir_from_dir(dir_path, any=True,object=object)
        usd_path = os.path.join(usd_dir_path, "mobility_new.usd")
        json_path = os.path.join(usd_dir_path, "link_handle_relationship.json")
        asset_init_pos = (0.75, 0, 0.6)
        cabinet_scale = (0.6, 0.6, 0.5)
        scene_key = "1"
        result["usd_path"] = usd_path
        result["json_path"] = json_path
        result["asset_init_pos"] = asset_init_pos
        result["scale"] = cabinet_scale # 이름을 scale로 통일
        result["scene_key"] = "1ext"
        result["guide_path"] = "1"
        with open(json_path, 'r') as f:
            json_info = json.load(f)
        hand_id = random.sample(json_info['handle'],1)[0]
        for key,value in json_info['relation'].items():
            if hand_id in value:
                link_id = key
                break
        if link_id in json_info['drawer']:
            body_type = 'drawer'
        elif link_id in json_info['cabinet']:
            body_type = 'cabinet'
        else:
            raise ValueError(f"link_id {link_id} is not in drawer or cabinet") 
        right_quarternion = np.array([0.5,0.5,-0.5,-0.5])
        joint_id = cabinet_get_joint_id_from_json(json_info, hand_id)
        min_link_point, max_link_point, center_link_point = link_points = np.array(json_info[f'points_{body_type}'][link_id])
        min_handle_point, max_handle_point, center_handle_point = hand_points = np.array(json_info[f'points_handle'][hand_id])
        link_pos_w = json_info['link_pos_w'][link_id]
        link_quat_w = json_info['link_quat_w'][link_id]
        link_mat_w = quat_to_matrix(link_quat_w)
        sampled_pos_guide = sample_point(hand_points,link_points)
        # sampled_pos_guide = np.array([sampled_pos_guide[i]*cabinet_scale[i] for i in range(3)])
        sampled_quat_guide = calculate_x_rotation_quaternion(sampled_pos_guide,center_handle_point)
        sampled_quat_guide = quaternion_multiply(sampled_quat_guide,right_quarternion)
        sampled_mat_guide = quat_to_matrix(sampled_quat_guide)
        sampled_pos_rel = np.transpose(link_mat_w) @ (sampled_pos_guide - link_pos_w)
        sampled_quat_rel = matrix_to_quat(np.transpose(link_mat_w) @ sampled_mat_guide)
        # sampled_pos_rel = sampled_pos_rel * np.array(cabinet_scale)

        #TODO:cabinet scale 차원 맞게 곱해지는지 확인
        handle_length = max_handle_point[1:] - min_handle_point[1:]
        handle_ratio = handle_length[1] / handle_length[0]
        if handle_ratio > 2:
            handle_type = "vertical"
        elif handle_ratio < 0.5:
            handle_type = "horizontal"
        else:
            handle_type = None
        result["handle_type"] = handle_type
        result["hand_id"] = hand_id
        result["joint_id"] = joint_id
        result["link_id"] = link_id
        result["link_points"] = [min_link_point*np.array(cabinet_scale), max_link_point*np.array(cabinet_scale), center_link_point*np.array(cabinet_scale)]
        result["handle_points"] = [min_handle_point*np.array(cabinet_scale), max_handle_point*np.array(cabinet_scale), center_handle_point*np.array(cabinet_scale)]
        result["action_list"] = [[min_handle_point*np.array(cabinet_scale), max_handle_point*np.array(cabinet_scale)],
                                 [min_link_point*np.array(cabinet_scale), max_link_point*np.array(cabinet_scale)]]
        result["guide_init_pos"] = sampled_pos_rel
        result["guide_init_quat"] = sampled_quat_rel
        result["joint_type"] = json_info["joint_body_connections"][joint_id]["joint_type"]
    if object == "door":
        usd_dir_path = sample_usd_dir_from_dir(dir_path, any=True,object=object)
        # select usd
        pull_mode = sysrand.choice([True, False])
        ccw_mode = sysrand.choice([True, False])
        # Guide mode
        guide_base_path = f"{ISAACLAB_ROOT_DIR}/Assets/guides/group_guide/guide_door_"
        # guide_mode = sysrand.randint(0,2)
        guide_mode = 0

        if not pull_mode and not ccw_mode:
            usd_path = os.path.join(usd_dir_path, "mobility_push_cw.usd")
            urdf_path = os.path.join(usd_dir_path, "mobility_push_cw.urdf")
            guide_str = "push_cw"
            scene_key = "3a"
        elif not pull_mode and ccw_mode:
            usd_path = os.path.join(usd_dir_path, "mobility_push_ccw.usd")
            urdf_path = os.path.join(usd_dir_path, "mobility_push_ccw.urdf")
            guide_str = "push_ccw"
            scene_key = "3b"
        elif pull_mode and not ccw_mode:
            usd_path = os.path.join(usd_dir_path, "mobility_pull_cw.usd")
            urdf_path = os.path.join(usd_dir_path, "mobility_pull_cw.urdf")
            guide_str = "pull_cw"
            scene_key = "3c"
        elif pull_mode and ccw_mode:
            usd_path = os.path.join(usd_dir_path, "mobility_pull_ccw.usd")
            urdf_path = os.path.join(usd_dir_path, "mobility_pull_ccw.urdf")
            guide_str = "pull_ccw"
            scene_key = "3d"
        bbox_path = os.path.join(usd_dir_path, "handle_bounding.json")
        guide_path = guide_base_path + guide_str + str(guide_mode) + ".usd"

        asset_init_pos = (0.4, 0.0, 0.915)
        door_scale = (0.9, 0.9, 0.9)

        joint2_x_position = load_urdf_and_get_joint2_x(urdf_path)
        guide_init_pos = (joint2_x_position+0.00, -0.03, 0.0)

        result["usd_path"] = usd_path
        result["guide_path"] = guide_path
        result["asset_init_pos"] = asset_init_pos
        result["guide_init_pos"] = guide_init_pos
        result["scale"] = door_scale # 이름을 scale로 통일
        result["pull_mode"] = pull_mode
        result["ccw_mode"] = ccw_mode
        result["scene_key"] = scene_key
    return result

def quaternion_multiply(q1, q2):
    """두 쿼터니언을 곱합니다 (q1 * q2)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def calculate_x_rotation_quaternion(p, q):
    """
    x=0 평면 위의 두 점 p, q를 잇는 벡터의 각도만큼
    x축을 기준으로 회전하는 쿼터니언(w, x, y, z)을 계산합니다.
    """

    # p에서 q로 가는 2D 벡터 (y, z 성분) 계산
    vector = np.array([q[1] - p[1], q[2] - p[2]])

    # 벡터가 0이면 회전이 필요 없으므로 단위 쿼터니언 반환
    if np.linalg.norm(vector) == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])

    # y축(양수)을 기준으로 벡터의 각도(theta)를 계산
    # arctan2(z, y)를 사용하여 올바른 사분면의 각도를 구함
    theta = np.arctan2(vector[1], vector[0])

    # x축을 기준으로 theta만큼 회전하는 쿼터니언 계산
    half_angle = theta / 2.0
    
    w = np.cos(half_angle)
    x = np.sin(half_angle) # 회전축이 x축(1, 0, 0)이므로 x 성분만 값을 가짐
    y = 0.0
    z = 0.0
    
    return np.array([w, x, y, z])
def quat_to_matrix(quat):
    w, x, y, z = quat
    matrix = np.array([[1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
                       [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
                       [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]])
    return matrix
def matrix_to_quat(matrix):
    """
    Converts a 3x3 rotation matrix to a quaternion [w, x, y, z].
    Handles all cases to ensure numerical stability.
    """
    trace = np.trace(matrix)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        # Determine which diagonal element is the largest
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
            
    return np.array([w, x, y, z])
def sample_point(handle_points:list,part_points:list, max_norm : float = 0.15):
    x = part_points[0][0]-0.03
    min_part = part_points[0][1:]
    min_part -= 0.03
    max_part = part_points[1][1:]
    max_part += 0.03
    handle_min, handle_max = handle_points[0][1:],handle_points[1][1:]
    handle_min -= 0.03
    handle_max += 0.03
    while True:
        yz_sample = np.random.uniform(min_part,max_part)
        in_handle = (handle_min <= yz_sample).all() and (yz_sample <= handle_max).all()
        if not in_handle and np.linalg.norm(yz_sample-handle_points[2][1:],1) < max_norm:
            return np.concatenate(([x],yz_sample))

def get_info_test(dir_path:str, object: str, asset_path: str, scene_key: str) -> Dict[str, Any]:
    result : Dict[str, Any] = {
        "usd_path": None, "guide_path": None, "json_path": None,
        "asset_init_pos": None, "guide_init_pos": None, "guide_init_quat": None,
        "scale": None, "guide_scale": None,
        "pull_mode": None, "ccw_mode": None, "close_mode": None,
        "scene_key": None, "cover_points": None,
        "joint2_x_position": None
    }

    if object == "bottle":
        if scene_key == "5a":
            close_mode = False
            guide_mode = 4
        elif scene_key == "5b":
            close_mode = False
            guide_mode = random.choice([0,2]) # 0
        elif scene_key == "5c":
            close_mode = False
            guide_mode = random.choice([1,3]) # 1
        elif scene_key == "5d":
            close_mode = True
            guide_mode = random.choice([0,2]) # 0
        elif scene_key == "5e":
            close_mode = True
            guide_mode = random.choice([1,3]) # 1
        elif scene_key == "5f":
            close_mode = True
            guide_mode = 4
        elif scene_key == "5g":
            close_mode = False
            guide_mode = 5
        elif scene_key == "5h":
            close_mode = True
            guide_mode = 5
        guide_base_path = f"{ISAACLAB_ROOT_DIR}/Assets/guides/group_guide/guide_mode_"
        guide_path = guide_base_path + str(guide_mode) + ".usd"

        reverse_mode = (guide_mode % 2) == 1
        usd_dir_path = os.path.join(dir_path,asset_path)
        if reverse_mode:
            usd_path = os.path.join(usd_dir_path, "mobility_reversed.usd")
        else:
            usd_path = os.path.join(usd_dir_path, "mobility.usd")
        bbox_path = os.path.join(usd_dir_path, "bounding_box.json")

        cover_points = load_bbox_and_compute_center_and_zmax(bbox_path, ["cover"])
        cx = cover_points["cover"]['center'][0]
        cy = cover_points["cover"]['center'][1]
        cz = cover_points["cover"]['center'][2] + 0.001

        asset_init_pos = (0.506, -0.02623, 0.5143)
        guide_init_pos = (cx, cy, cover_points["cover"]["max"][2])

        # compute guide scale from bbox
        xy_scale = 8.5 * (cover_points['cover']['max'] - cover_points['cover']['min'])
        xy_scale[-1] = 0.8
        guide_scale = tuple(xy_scale)

        result["usd_path"] = usd_path
        result["guide_path"] = guide_path
        result["asset_init_pos"] = asset_init_pos
        result["guide_init_pos"] = guide_init_pos
        result["guide_scale"] = guide_scale
        result["close_mode"] = close_mode
        result["scene_key"] = scene_key
        result["cover_points"] = cover_points

    if object == "cabinet":
        usd_dir_path = os.path.join(dir_path, asset_path)
        usd_path = os.path.join(usd_dir_path, "mobility_new.usd")
        json_path = os.path.join(usd_dir_path, "link_handle_relationship.json")
        asset_init_pos = (0.75, 0, 0.6)
        cabinet_scale = (0.5, 0.5, 0.5)#0.6, 0.6, 0.5
        result["usd_path"] = usd_path
        result["json_path"] = json_path
        result["asset_init_pos"] = asset_init_pos
        result["scale"] = cabinet_scale # 이름을 scale로 통일
        result["scene_key"] = "1ext"
        result["guide_path"] = "1"
        with open(json_path, 'r') as f:
            json_info = json.load(f)
        hand_id = json_info['handle'][int(scene_key)]
        for key,value in json_info['relation'].items():
            if hand_id in value:
                link_id = key
                break
        if link_id in json_info['drawer']:
            body_type = 'drawer'
        elif link_id in json_info['cabinet']:
            body_type = 'cabinet'
        else:
            raise ValueError(f"link_id {link_id} is not in drawer or cabinet") 
        right_quarternion = np.array([0.5,0.5,-0.5,-0.5])
        joint_id = cabinet_get_joint_id_from_json(json_info, hand_id)
        min_link_point, max_link_point, center_link_point = link_points = np.array(json_info[f'points_{body_type}'][link_id])
        min_handle_point, max_handle_point, center_handle_point = hand_points = np.array(json_info[f'points_handle'][hand_id])
        link_pos_w = json_info['link_pos_w'][link_id]
        link_quat_w = json_info['link_quat_w'][link_id]
        link_mat_w = quat_to_matrix(link_quat_w)
        sampled_pos_guide = sample_point(hand_points,link_points)
        # sampled_pos_guide = np.array([sampled_pos_guide[i]*cabinet_scale[i] for i in range(3)])
        sampled_quat_guide = calculate_x_rotation_quaternion(sampled_pos_guide,center_handle_point)
        sampled_quat_guide = quaternion_multiply(sampled_quat_guide,right_quarternion)
        sampled_mat_guide = quat_to_matrix(sampled_quat_guide)
        sampled_pos_rel = np.transpose(link_mat_w) @ (sampled_pos_guide - link_pos_w)
        sampled_quat_rel = matrix_to_quat(np.transpose(link_mat_w) @ sampled_mat_guide)
        # sampled_pos_rel = sampled_pos_rel * np.array(cabinet_scale)
        result["hand_id"] = hand_id
        result["joint_id"] = joint_id
        result["link_id"] = link_id
        result["link_points"] = [min_link_point*np.array(cabinet_scale), max_link_point*np.array(cabinet_scale), center_link_point*np.array(cabinet_scale)]
        result["handle_points"] = [min_handle_point*np.array(cabinet_scale), max_handle_point*np.array(cabinet_scale), center_handle_point*np.array(cabinet_scale)]
        result["action_list"] = [[min_handle_point*np.array(cabinet_scale), max_handle_point*np.array(cabinet_scale)],
                                 [min_link_point*np.array(cabinet_scale), max_link_point*np.array(cabinet_scale)]]
        result["guide_init_pos"] = sampled_pos_rel
        result["guide_init_quat"] = sampled_quat_rel
        result["joint_type"] = json_info["joint_body_connections"][joint_id]["joint_type"]
    if object == "door":
        usd_dir_path = os.path.join(dir_path, asset_path)
        # Guide mode
        guide_base_path = f"{ISAACLAB_ROOT_DIR}/Assets/guides/group_guide/guide_door_"
        # guide_mode = sysrand.randint(0,2)
        guide_mode = 0        
        if scene_key == "3a":
            pull_mode = False
            ccw_mode = False
            usd_path = os.path.join(usd_dir_path, "mobility_push_cw.usd")
            urdf_path = os.path.join(usd_dir_path, "mobility_push_cw.urdf")
            guide_str = "push_cw"
        elif scene_key == "3b":
            pull_mode = False
            ccw_mode = True
            usd_path = os.path.join(usd_dir_path, "mobility_push_ccw.usd")
            urdf_path = os.path.join(usd_dir_path, "mobility_push_ccw.urdf")
            guide_str = "push_ccw"
        elif scene_key == "3c":
            pull_mode = True
            ccw_mode = False
            usd_path = os.path.join(usd_dir_path, "mobility_pull_cw.usd")
            urdf_path = os.path.join(usd_dir_path, "mobility_pull_cw.urdf")
            guide_str = "pull_cw"
        elif scene_key == "3d":
            pull_mode = True
            ccw_mode = True
            usd_path = os.path.join(usd_dir_path, "mobility_pull_ccw.usd")
            urdf_path = os.path.join(usd_dir_path, "mobility_pull_ccw.urdf")
            guide_str = "pull_ccw"

        bbox_path = os.path.join(usd_dir_path, "handle_bounding.json")
        guide_path = guide_base_path + guide_str + str(guide_mode) + ".usd"

        asset_init_pos = (0.4, 0.0, 0.915)
        door_scale = (0.9, 0.9, 0.9)

        joint2_x_position = load_urdf_and_get_joint2_x(urdf_path)
        guide_init_pos = (joint2_x_position, -0.04, 0.0)

        result["usd_path"] = usd_path
        result["guide_path"] = guide_path
        result["asset_init_pos"] = asset_init_pos
        result["guide_init_pos"] = guide_init_pos
        result["scale"] = door_scale # 이름을 scale로 통일
        result["pull_mode"] = pull_mode
        result["ccw_mode"] = ccw_mode
        result["scene_key"] = scene_key
    return result


def get_info_collect(dir_path:str, object: str, asset_path: str, scene_key: str) -> Dict[str, Any]:
    result : Dict[str, Any] = {
        "usd_path": None, "guide_path": None, "json_path": None,
        "asset_init_pos": None, "guide_init_pos": None, "guide_init_quat": None,
        "scale": None, "guide_scale": None,
        "pull_mode": None, "ccw_mode": None, "close_mode": None,
        "scene_key": None, "cover_points": None,
        "joint2_x_position": None
    }

    if object == "bottle":
        if scene_key == "5a":
            close_mode = False
            guide_mode = 4
        elif scene_key == "5b":
            close_mode = False
            guide_mode = random.choice([0,2]) # 0
        elif scene_key == "5c":
            close_mode = False
            guide_mode = random.choice([1,3]) # 1
        elif scene_key == "5d":
            close_mode = True
            guide_mode = random.choice([0,2]) # 0
        elif scene_key == "5e":
            close_mode = True
            guide_mode = random.choice([1,3]) # 1
        elif scene_key == "5f":
            close_mode = True
            guide_mode = 4
        elif scene_key == "5g":
            close_mode = False
            guide_mode = 5
        elif scene_key == "5h":
            close_mode = True
            guide_mode = 5
        guide_base_path = f"{ISAACLAB_ROOT_DIR}/Assets/guides/group_guide/guide_mode_"
        guide_path = guide_base_path + str(guide_mode) + ".usd"

        reverse_mode = (guide_mode % 2) == 1
        usd_dir_path = os.path.join(dir_path,asset_path)
        if reverse_mode:
            usd_path = os.path.join(usd_dir_path, "mobility_reversed.usd")
        else:
            usd_path = os.path.join(usd_dir_path, "mobility.usd")
        bbox_path = os.path.join(usd_dir_path, "bounding_box.json")

        cover_points = load_bbox_and_compute_center_and_zmax(bbox_path, ["cover"])
        cx = cover_points["cover"]['center'][0]
        cy = cover_points["cover"]['center'][1]
        cz = cover_points["cover"]['center'][2] + 0.001

        asset_init_pos = (0.506, -0.02623, 0.5143)
        guide_init_pos = (cx, cy, cover_points["cover"]["max"][2])

        # compute guide scale from bbox
        xy_scale = 8.5 * (cover_points['cover']['max'] - cover_points['cover']['min'])
        xy_scale[-1] = 0.8
        guide_scale = tuple(xy_scale)

        result["usd_path"] = usd_path
        result["guide_path"] = guide_path
        result["asset_init_pos"] = asset_init_pos
        result["guide_init_pos"] = guide_init_pos
        result["guide_scale"] = guide_scale
        result["close_mode"] = close_mode
        result["scene_key"] = scene_key
        result["cover_points"] = cover_points

    if object == "cabinet":
        usd_dir_path = os.path.join(dir_path, asset_path)
        usd_path = os.path.join(usd_dir_path, "mobility_new.usd")
        json_path = os.path.join(usd_dir_path, "link_handle_relationship.json")
        asset_init_pos = (0.75, 0, 0.6)
        cabinet_scale = (0.6, 0.6, 0.5)
        result["usd_path"] = usd_path
        result["json_path"] = json_path
        result["asset_init_pos"] = asset_init_pos
        result["scale"] = cabinet_scale # 이름을 scale로 통일
        result["scene_key"] = "1ext"
        result["guide_path"] = "1"
        with open(json_path, 'r') as f:
            json_info = json.load(f)
        hand_id = resolve_cabinet_collect_handle_id(asset_path, scene_key, json_info, json_path)
        link_id = cabinet_get_link_id_from_json(json_info, hand_id)
        if link_id in json_info['drawer']:
            body_type = 'drawer'
        elif link_id in json_info['cabinet']:
            body_type = 'cabinet'
        else:
            raise ValueError(f"link_id {link_id} is not in drawer or cabinet") 
        right_quarternion = np.array([0.5,0.5,-0.5,-0.5])
        joint_id = cabinet_get_joint_id_from_json(json_info, hand_id)
        min_link_point, max_link_point, center_link_point = link_points = np.array(json_info[f'points_{body_type}'][link_id])
        min_handle_point, max_handle_point, center_handle_point = hand_points = np.array(json_info[f'points_handle'][hand_id])
        link_pos_w = json_info['link_pos_w'][link_id]
        link_quat_w = json_info['link_quat_w'][link_id]
        link_mat_w = quat_to_matrix(link_quat_w)
        sampled_pos_guide = sample_point(hand_points,link_points)
        # sampled_pos_guide = np.array([sampled_pos_guide[i]*cabinet_scale[i] for i in range(3)])
        sampled_quat_guide = calculate_x_rotation_quaternion(sampled_pos_guide,center_handle_point)
        sampled_quat_guide = quaternion_multiply(sampled_quat_guide,right_quarternion)
        sampled_mat_guide = quat_to_matrix(sampled_quat_guide)
        sampled_pos_rel = np.transpose(link_mat_w) @ (sampled_pos_guide - link_pos_w)
        sampled_quat_rel = matrix_to_quat(np.transpose(link_mat_w) @ sampled_mat_guide)
        # sampled_pos_rel = sampled_pos_rel * np.array(cabinet_scale)
        handle_length = max_handle_point[1:] - min_handle_point[1:]
        handle_ratio = handle_length[1] / handle_length[0]
        if handle_ratio > 2:
            handle_type = "vertical"
        elif handle_ratio < 0.5:
            handle_type = "horizontal"
        else:
            handle_type = None
        result["handle_type"] = handle_type
        result["hand_id"] = hand_id
        result["joint_id"] = joint_id
        result["link_id"] = link_id
        result["link_points"] = [min_link_point*np.array(cabinet_scale), max_link_point*np.array(cabinet_scale), center_link_point*np.array(cabinet_scale)]
        result["handle_points"] = [min_handle_point*np.array(cabinet_scale), max_handle_point*np.array(cabinet_scale), center_handle_point*np.array(cabinet_scale)]
        result["action_list"] = [[min_handle_point*np.array(cabinet_scale), max_handle_point*np.array(cabinet_scale)],
                                 [min_link_point*np.array(cabinet_scale), max_link_point*np.array(cabinet_scale)]]
        result["guide_init_pos"] = sampled_pos_rel
        result["guide_init_quat"] = sampled_quat_rel
        result["joint_type"] = json_info["joint_body_connections"][joint_id]["joint_type"]
    if object == "door":
        usd_dir_path = os.path.join(dir_path, asset_path)
        # Guide mode
        guide_base_path = f"{ISAACLAB_ROOT_DIR}/Assets/guides/group_guide/guide_door_"
        # guide_mode = sysrand.randint(0,2)
        guide_mode = 0        
        if scene_key == "3a":
            pull_mode = False
            ccw_mode = False
            usd_path = os.path.join(usd_dir_path, "mobility_push_cw.usd")
            urdf_path = os.path.join(usd_dir_path, "mobility_push_cw.urdf")
            guide_str = "push_cw"
        elif scene_key == "3b":
            pull_mode = False
            ccw_mode = True
            usd_path = os.path.join(usd_dir_path, "mobility_push_ccw.usd")
            urdf_path = os.path.join(usd_dir_path, "mobility_push_ccw.urdf")
            guide_str = "push_ccw"
        elif scene_key == "3c":
            pull_mode = True
            ccw_mode = False
            usd_path = os.path.join(usd_dir_path, "mobility_pull_cw.usd")
            urdf_path = os.path.join(usd_dir_path, "mobility_pull_cw.urdf")
            guide_str = "pull_cw"
        elif scene_key == "3d":
            pull_mode = True
            ccw_mode = True
            usd_path = os.path.join(usd_dir_path, "mobility_pull_ccw.usd")
            urdf_path = os.path.join(usd_dir_path, "mobility_pull_ccw.urdf")
            guide_str = "pull_ccw"

        bbox_path = os.path.join(usd_dir_path, "handle_bounding.json")
        guide_path = guide_base_path + guide_str + str(guide_mode) + ".usd"

        asset_init_pos = (0.4, 0.0, 0.915)
        door_scale = (0.9, 0.9, 0.9)

        joint2_x_position = load_urdf_and_get_joint2_x(urdf_path)
        guide_init_pos = (joint2_x_position+0.00, -0.03, 0.0)

        result["usd_path"] = usd_path
        result["guide_path"] = guide_path
        result["asset_init_pos"] = asset_init_pos
        result["guide_init_pos"] = guide_init_pos
        result["scale"] = door_scale # 이름을 scale로 통일
        result["pull_mode"] = pull_mode
        result["ccw_mode"] = ccw_mode
        result["scene_key"] = scene_key
    return result
