import os
import glob
import xml.etree.ElementTree as ET

def process_urdf_file(orig_path: str, out_name: str = "mobility_reversed.urdf"):
    """
    revolute joint의 axis xyz를 "0 0 -1"로 강제 변경하고 결과를 새 파일로 저장.
    """
    tree = ET.parse(orig_path)
    root = tree.getroot()
    namespace = ""  # URDF usually has no xmlns, but if there is, you'd need to handle it.

    changed = False
    for joint in root.findall("joint"):
        joint_type = joint.get("type", "")
        if joint_type == "revolute":
            axis = joint.find("axis")
            if axis is None:
                # 없으면 새로 만들어 붙인다
                axis = ET.SubElement(joint, "axis")
            # 항상 xyz 속성을 "0 0 -1"로 설정
            prev = axis.get("xyz")
            axis.set("xyz", "0 0 -1")
            if prev != "0 0 -1":
                changed = True

    if not changed:
        # 그래도 저장은 한다 (원래도 같을 수 있음)
        pass

    out_path = os.path.join(os.path.dirname(orig_path), out_name)
    # pretty print: ElementTree doesn't indent by default; simple workaround
    indent_xml(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Processed '{orig_path}' -> '{out_path}' (revolute axes set to 0 0 -1)")

def indent_xml(elem, level=0):
    """간단한 pretty-print 들여쓰기 (ElementTree용)."""
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level+1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def traverse_and_process(bottle_root: str):
    """
    bottle_root 안의 모든 하위 디렉터리를 뒤져서 *.urdf 파일을 찾아 처리.
    """
    # 예: /.../bottle/*/mobility.urdf
    for sub in os.listdir(bottle_root):
        subdir = os.path.join(bottle_root, sub)
        if not os.path.isdir(subdir):
            continue
        # 모든 .urdf 파일 (유연하게)
        urdf_paths = glob.glob(os.path.join(subdir, "*.urdf"))
        for urdf in urdf_paths:
            process_urdf_file(urdf)

if __name__ == "__main__":
    bottle_root = "Assets/AdaManip/bottle"
    traverse_and_process(bottle_root)
