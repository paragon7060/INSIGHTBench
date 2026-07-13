import os
import xml.etree.ElementTree as ET
import copy
import argparse

def modify_xyz_value(xyz_str, new_y):
    """기존 함수: y 좌표 값을 새로운 값으로 설정합니다."""
    xyz = xyz_str.strip().split()
    if len(xyz) != 3:
        return xyz_str
    xyz[1] = str(new_y)
    return " ".join(xyz)

def modify_xyz_x_value(xyz_str, delta_x):
    """새로운 함수: x 좌표 값에서 delta_x 만큼 뺍니다."""
    xyz = xyz_str.strip().split()
    if len(xyz) != 3:
        return xyz_str
    try:
        # x 좌표를 float으로 변환하여 값을 뺀 후 다시 문자열로 변환합니다.
        new_x = float(xyz[0]) - delta_x
        xyz[0] = str(new_x)
    except ValueError:
        # 숫자 변환에 실패하면 원본 문자열을 반환합니다.
        return xyz_str
    return " ".join(xyz)

def modify_urdf_tree(tree: ET.ElementTree, joint1_axis: str, joint2_axis: str) -> ET.ElementTree:
    root = tree.getroot()

    for link in root.findall('link'):
        if link.attrib.get('name') == 'link_1':
            for tag in link.findall('visual') + link.findall('collision'):
                origin = tag.find('origin')
                if origin is not None:
                    origin.set('xyz', modify_xyz_value(origin.attrib.get('xyz', '0 0 0'), 0.0))

    for joint in root.findall('joint'):
        name = joint.attrib.get('name')
        if name == 'joint_1':
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', modify_xyz_value(origin.attrib.get('xyz', '0 0 0'), 0.0))
            axis = joint.find('axis')
            if axis is not None:
                axis.set('xyz', joint1_axis)

        elif name == 'joint_2':
            origin = joint.find('origin')
            if origin is not None:
                # 1. 원본 xyz 값을 가져옵니다.
                current_xyz = origin.attrib.get('xyz', '0 0 0')
                # 2. y 좌표 값을 -0.06으로 설정합니다.
                xyz_with_new_y = modify_xyz_value(current_xyz, -0.06)
                # 3. 그 결과에서 x 좌표 값을 줄입니다.
                final_xyz = modify_xyz_x_value(xyz_with_new_y, 0.05)
                # 4. 최종적으로 수정된 xyz 값을 설정합니다.
                origin.set('xyz', final_xyz)
            axis = joint.find('axis')
            if axis is not None:
                axis.set('xyz', joint2_axis)

    return tree

def batch_generate_urdfs(base_dir):
    joint1_options = {
        'push': '0 0 1',
        'pull': '0 0 -1',
    }
    joint2_options = {
        'cw': '0 1 0',
        'ccw': '0 -1 0',
    }

    for root_dir, _, files in os.walk(base_dir):
        if 'mobility.urdf' in files:
            input_path = os.path.join(root_dir, 'mobility.urdf')
            try:
                base_tree = ET.parse(input_path)

                for j1_tag, j1_axis in joint1_options.items():
                    for j2_tag, j2_axis in joint2_options.items():
                        new_tree = modify_urdf_tree(copy.deepcopy(base_tree), j1_axis, j2_axis)
                        output_name = f"mobility_{j1_tag}_{j2_tag}.urdf"
                        output_path = os.path.join(root_dir, output_name)
                        new_tree.write(output_path, encoding='utf-8', xml_declaration=True)
                        print(f"Generated: {output_path}")

            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

def batch_generate_urdfs_for_dirs(base_dirs):
    for base_dir in base_dirs:
        print(f"\n=== Processing base_dir: {base_dir} ===")
        if not os.path.isdir(base_dir):
            print(f"Skip (not a directory): {base_dir}")
            continue
        batch_generate_urdfs(base_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate URDF variants across multiple base directories.")
    parser.add_argument(
        "base_dirs",
        nargs="*",
        default=[
            "Assets/AdaManip/door",
            "Assets/UniDoorManip/Datasets/LeverDoor"
            ],
        )
    args = parser.parse_args()
    batch_generate_urdfs_for_dirs(args.base_dirs)
