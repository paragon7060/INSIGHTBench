#!/usr/bin/env python3
# modify_urdf_axis.py

"""
Recursively traverse a folder and set all <axis> xyz attributes to "0 0 1" for every .urdf file,
and adjust specific joint limits:
  - joint_1: upper="0.005"
  - joint_2: upper="6.28"
Usage:
    python change_axis.py --folder path/to/urdfs [-o path/to/output_folder]

- If no output folder is given, original .urdf files are overwritten in-place.
- If an output folder is provided, the script recreates the directory structure under it and writes modified files there.
"""
import xml.etree.ElementTree as ET
import argparse
import os
import sys


def modify_urdf(input_path: str, output_path: str):
    """Modify a single URDF file: set axis xyz, and update joint limits."""
    try:
        tree = ET.parse(input_path)
    except ET.ParseError as e:
        print(f"[ERROR] Failed to parse {input_path}: {e}")
        return False

    root = tree.getroot()
    changed = False

    # 1) Update all <axis> xyz to "0 0 1"
    for axis in root.findall('.//axis'):
        if axis.get('xyz') != '0 0 1':
            axis.set('xyz', '0 0 1')
            changed = True

    # 2) Update specific joint limits
    for joint in root.findall('.//joint'):
        name = joint.get('name', '')
        limit = joint.find('limit')
        if limit is None:
            continue
        if name == 'joint_1':
            if limit.get('upper') != '0.005':
                limit.set('upper', '0.005')
                changed = True
                print(f"[LIMIT] Set upper=0.005 for joint 'joint_1'")
        elif name == 'joint_2':
            if limit.get('upper') != '6.28':
                limit.set('upper', '6.28')
                changed = True
                print(f"[LIMIT] Set upper=6.28 for joint 'joint_2'")

    # Write changes
    if changed:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"[MODIFIED] {input_path} -> {output_path}")
    else:
        if input_path != output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            print(f"[COPIED] {input_path} -> {output_path} (no changes needed)")
        else:
            print(f"[SKIPPED] {input_path} (already up-to-date)")
    return True


def process_folder(base_folder: str, output_folder: str = None):
    """Walk through base_folder to find .urdf files and process each."""
    for root, dirs, files in os.walk(base_folder):
        for fname in files:
            if fname.lower().endswith('.urdf'):
                input_path = os.path.join(root, fname)
                if output_folder:
                    rel = os.path.relpath(input_path, base_folder)
                    output_path = os.path.join(output_folder, rel)
                else:
                    output_path = input_path
                modify_urdf(input_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch modify axis XYZ and joint limits in URDF files under a folder.'
    )
    parser.add_argument(
        '--folder', '-f',
        default="Assets/AdaManip/bottle",
        help='Path to folder containing URDF files to process.'
    )
    parser.add_argument(
        '-o', '--output',
        help='Optional output folder. If provided, modified files are written here preserving directory structure.'
    )
    args = parser.parse_args()

    base = args.folder
    out = args.output

    if not os.path.isdir(base):
        sys.exit(f"Error: '{base}' is not a valid directory.")

    if out and not os.path.exists(out):
        os.makedirs(out, exist_ok=True)

    process_folder(base, out)