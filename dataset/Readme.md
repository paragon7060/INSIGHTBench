# To find corrupted videos
python dataset/find_corrupted_video.py "paragon7060/INSIGHT-data-noguide-1" "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHT-data-noguide-1"
## guide
python dataset/find_corrupted_video.py "paragon7060/INSIGHTfixpos" "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTfixpos"
python dataset/find_corrupted_video.py "paragon7060/INSIGHTposrand" "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTposrand"
python dataset/find_corrupted_video.py "paragon7060/INSIGHTposrand_final" "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTposrand_final"
## noguide
python dataset/find_corrupted_video.py "paragon7060/INSIGHTfixpos-noguide" "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTfixpos-noguide"
## param
python dataset/find_corrupted_video.py "paragon7060/INSIGHTfixpos-param" "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTparam"
## param-posrand
python dataset/find_corrupted_video.py "paragon7060/INSIGHTparam-posrand" "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTparam-posrand"

# To merge datasets
!!! Merge has error with action size!
'''
python dataset/dataset_merge.py --sources "dataset1_path" "dataset2_path" --output "dataset_output_dir"
'''
## guide
'''
python dataset/dataset_merge.py --sources "data/paragon7060/INSIGHT-fixpos-guide-0" "data/paragon7060/INSIGHT-fixpos-guide-1" "data/paragon7060/INSIGHT-fixpos-guide-2" "data/paragon7060/INSIGHT-fixpos-guide-3" --output "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTfixpos"
'''
### posrand
'''
python dataset/dataset_merge.py --sources "data/paragon7060/INSIGHT-posrand-guide-0" "data/paragon7060/INSIGHT-posrand-guide-1" "data/paragon7060/INSIGHT-posrand-guide-2" "data/paragon7060/INSIGHT-posrand-guide-3" --output "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTposrand"
'''
'''
python dataset/dataset_merge.py --sources "data/paragon7060/INSIGHTposrand" "data/paragon7060/INSIGHTposrand_color" --output "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTposrand_final"
'''
check action size
## noguide
'''
python dataset/dataset_merge.py --sources "data/paragon7060/INSIGHT-fixpos-no_guide-0" "data/paragon7060/INSIGHT-fixpos-no_guide-1" "data/paragon7060/INSIGHT-fixpos-no_guide-2" "data/paragon7060/INSIGHT-fixpos-no_guide-3"  --output "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTfixpos-noguide"
'''
check action size
### posrand
'''
python dataset/dataset_merge.py --sources "data/paragon7060/INSIGHT-posrand-no_guide-0" "data/paragon7060/INSIGHT-posrand-no_guide-1" "data/paragon7060/INSIGHT-posrand-no_guide-2" "data/paragon7060/INSIGHT-posrand-no_guide-3" --output "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTposrand_noguide"
'''
### param
'''
python dataset/dataset_merge.py --sources "data/paragon7060/INSIGHT-param-guide-0" "data/paragon7060/INSIGHT-param-guide-1" "data/paragon7060/INSIGHT-param-guide-2" "data/paragon7060/INSIGHT-param-guide-3" --output "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTparam"
'''
### param posrand + color ...
'''
python dataset/dataset_merge.py --sources "data/paragon7060/INSIGHT-param-posrand-color-guide-0" "data/paragon7060/INSIGHT-param-posrand-color-guide-1" "data/paragon7060/INSIGHT-param-posrand-color-guide-2" "data/paragon7060/INSIGHT-param-posrand-color-guide-3" "data/paragon7060/INSIGHT-param-posrand-guide-0" "data/paragon7060/INSIGHT-param-posrand-guide-1" "data/paragon7060/INSIGHT-param-posrand-guide-2" "data/paragon7060/INSIGHT-param-posrand-guide-3" --output "/home/seonho/workspace/IsaacLab/data/paragon7060/INSIGHTparam-posrand"
'''

# To check dataset distribution
'''
python dataset/analyze_dataset_tasks_with_total.py 
'''

# tar gz
tar -zcvf INSIGHTv2.tar.gz data/paragon7060/INSIGHTv2
tar -zcvf INSIGHT-NG-v2.tar.gz data/paragon7060/INSIGHT-NG-v2


# Send to Server
scp -r fixpos.tar.gz seonho@166.104.35.48:/home/seonho/workspace/INSIGHTfixpos.tar.gz
scp -r fixpos_noguide.tar.gz seonho@166.104.35.48:/home/seonho/workspace/INSIGHTfixpos-ng.tar.gz

scp -r fixpos.tar.gz junhyeong@166.104.35.50:/home/junhyeong/workspace/INSIGHTfixpos.tar.gz
scp -r fixpos_noguide.tar.gz junhyeong@166.104.35.50:/home/junhyeong/workspace/INSIGHTfixpos-ng.tar.gz
