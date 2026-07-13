# Dataset Utilities

Set `DATA_ROOT` to the directory containing your local datasets before running
these examples:

```bash
export DATA_ROOT=/path/to/datasets
```

## Find Corrupted Videos

```bash
python dataset/find_corrupted_video.py "paragon7060/INSIGHT-data-noguide-1" "${DATA_ROOT}/paragon7060/INSIGHT-data-noguide-1"

# Guide datasets
python dataset/find_corrupted_video.py "paragon7060/INSIGHTfixpos" "${DATA_ROOT}/paragon7060/INSIGHTfixpos"
python dataset/find_corrupted_video.py "paragon7060/INSIGHTposrand" "${DATA_ROOT}/paragon7060/INSIGHTposrand"
python dataset/find_corrupted_video.py "paragon7060/INSIGHTposrand_final" "${DATA_ROOT}/paragon7060/INSIGHTposrand_final"

# No-guide and parameter datasets
python dataset/find_corrupted_video.py "paragon7060/INSIGHTfixpos-noguide" "${DATA_ROOT}/paragon7060/INSIGHTfixpos-noguide"
python dataset/find_corrupted_video.py "paragon7060/INSIGHTfixpos-param" "${DATA_ROOT}/paragon7060/INSIGHTparam"
python dataset/find_corrupted_video.py "paragon7060/INSIGHTparam-posrand" "${DATA_ROOT}/paragon7060/INSIGHTparam-posrand"
```

## Merge Datasets

Verify compatible action dimensions before merging datasets.

```bash
python dataset/dataset_merge.py \
  --sources "${DATA_ROOT}/paragon7060/INSIGHT-fixpos-guide-0" "${DATA_ROOT}/paragon7060/INSIGHT-fixpos-guide-1" "${DATA_ROOT}/paragon7060/INSIGHT-fixpos-guide-2" "${DATA_ROOT}/paragon7060/INSIGHT-fixpos-guide-3" \
  --output "${DATA_ROOT}/paragon7060/INSIGHTfixpos"

python dataset/dataset_merge.py \
  --sources "${DATA_ROOT}/paragon7060/INSIGHT-posrand-guide-0" "${DATA_ROOT}/paragon7060/INSIGHT-posrand-guide-1" "${DATA_ROOT}/paragon7060/INSIGHT-posrand-guide-2" "${DATA_ROOT}/paragon7060/INSIGHT-posrand-guide-3" \
  --output "${DATA_ROOT}/paragon7060/INSIGHTposrand"

python dataset/dataset_merge.py \
  --sources "${DATA_ROOT}/paragon7060/INSIGHTposrand" "${DATA_ROOT}/paragon7060/INSIGHTposrand_color" \
  --output "${DATA_ROOT}/paragon7060/INSIGHTposrand_final"

python dataset/dataset_merge.py \
  --sources "${DATA_ROOT}/paragon7060/INSIGHT-fixpos-no_guide-0" "${DATA_ROOT}/paragon7060/INSIGHT-fixpos-no_guide-1" "${DATA_ROOT}/paragon7060/INSIGHT-fixpos-no_guide-2" "${DATA_ROOT}/paragon7060/INSIGHT-fixpos-no_guide-3" \
  --output "${DATA_ROOT}/paragon7060/INSIGHTfixpos-noguide"

python dataset/dataset_merge.py \
  --sources "${DATA_ROOT}/paragon7060/INSIGHT-posrand-no_guide-0" "${DATA_ROOT}/paragon7060/INSIGHT-posrand-no_guide-1" "${DATA_ROOT}/paragon7060/INSIGHT-posrand-no_guide-2" "${DATA_ROOT}/paragon7060/INSIGHT-posrand-no_guide-3" \
  --output "${DATA_ROOT}/paragon7060/INSIGHTposrand_noguide"

python dataset/dataset_merge.py \
  --sources "${DATA_ROOT}/paragon7060/INSIGHT-param-guide-0" "${DATA_ROOT}/paragon7060/INSIGHT-param-guide-1" "${DATA_ROOT}/paragon7060/INSIGHT-param-guide-2" "${DATA_ROOT}/paragon7060/INSIGHT-param-guide-3" \
  --output "${DATA_ROOT}/paragon7060/INSIGHTparam"

python dataset/dataset_merge.py \
  --sources "${DATA_ROOT}/paragon7060/INSIGHT-param-posrand-color-guide-0" "${DATA_ROOT}/paragon7060/INSIGHT-param-posrand-color-guide-1" "${DATA_ROOT}/paragon7060/INSIGHT-param-posrand-color-guide-2" "${DATA_ROOT}/paragon7060/INSIGHT-param-posrand-color-guide-3" "${DATA_ROOT}/paragon7060/INSIGHT-param-posrand-guide-0" "${DATA_ROOT}/paragon7060/INSIGHT-param-posrand-guide-1" "${DATA_ROOT}/paragon7060/INSIGHT-param-posrand-guide-2" "${DATA_ROOT}/paragon7060/INSIGHT-param-posrand-guide-3" \
  --output "${DATA_ROOT}/paragon7060/INSIGHTparam-posrand"
```

## Analyze and Package Datasets

```bash
python dataset/analyze_dataset_tasks_with_total.py
python dataset/calculate_video_frame.py "${DATA_ROOT}/paragon7060/dataset-name/videos/chunk-000"

tar -zcvf INSIGHTv2.tar.gz "${DATA_ROOT}/paragon7060/INSIGHTv2"
tar -zcvf INSIGHT-NG-v2.tar.gz "${DATA_ROOT}/paragon7060/INSIGHT-NG-v2"
```

## Transfer an Archive

Use a destination you control; do not place account names, hosts, or private
paths in this repository.

```bash
scp -r archive.tar.gz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"
```
