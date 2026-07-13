# For PartManip Cabinet Preprocessing
Asset folder : Assets/PartManip/drawer/
## 1. Download and move the asset files
1. Download PartManip Dataset
2. we use only drawer dataset
3. move the assets to the Assets folder
## 2. USD convert
'''
unzip Assets/PartManip/drawer.zip
bash urdf_convert_ext.sh drawer
'''

# For Adamanip Bottle URDF Preprocessing
Asset folder : Assets/AdaManip/assets/...
## 1. URDF axis change 
'''
python assets/asset_lab/urdf_tools/change_axis.py
'''
## 2. Make URDF with reversed axis
'''
python assets/asset_lab/urdf_tools/change_joint2_axis_reverse.py
'''
## 3. Make USD files for URDF files
in IsaacLab,
'''
bash urdf_convert_ext.sh bottle
'''

## Changes in URDF
'''
12b : {"min": [-0.0358883372827304, -0.03344405638455951, -0.14676703800594484], "max": [0.034111662717269606, 0.036555900227699716, 0.01828898560265311], "cover_min": [-0.0358883372827304, -0.03321800418904833, -0.011330378765173638], "cover_max": [0.034111662717269606, 0.036555900227699716, 0.01728898560265311], "body_min": [-0.0358883372827304, -0.03344405638455951, -0.14676703800594484], "body_max": [0.034111662717269606, 0.036555900227699716, -0.011330378765173638]}
b26 : delete
b27 : delete
b28 : delete
b29 : delete
b30 : delete
b31 : {"body_min": [-0.037609006709808004, -0.037609006709808004, -0.0944765523637571], "body_max": [0.037609012871885665, 0.037609012871885665, 0.0077953415018237925], "cover_min": [-0.03767865532705078, -0.03767865532705078, -0.012481029303379337], "cover_max": [0.03767861867076982, 0.03767861867076982, 0.015680449689096915], "min": [-0.037609006709808004, -0.037609006709808004, -0.0944765523637571], "max": [0.037609012871885665, 0.037609012871885665, 0.012680449689096915]}
'''

# For Adamanip Door URDF Preprocessing
## 1. Change URDF (fix position or axis)
'''
python assets/asset_lab/urdf_tools/change_door.py
'''
## 2. Make USD for each URDF
in IsaacLab folder,
'''
bash assets/asset_lab/urdf_convert_ext.sh door
'''

# For Cabinet assets
## 1. Unzip PartManip dataset
## 2. urdf -> usd export
'''
bash assets/asset_lab/urdf_convert_ext.sh cabinet
'''