#!/bin/bash
set -x
set -e

DATASET_TYPE="NYU_VAL"
MODEL_TYPE="astrous-os8-concat"

if [ "$DATASET_TYPE" = "NYU_VAL" ]; then 
	echo nyu_validation dataset
	image_dir="/PATH_TO_NYU/nyu_label_image/nyu_depth/"
	pred_save_dir="/PATH_TO_NYU/adenet_predict/"$MODEL_TYPE"/"
	rgb_extend="_colors.png"
fi

evaluate_depth_npy=$pred_save_dir"depths_npy/"

if [ "$MODEL_TYPE" = "astrous-os8-concat" ]; then
	echo model_type is astrous_os8-concat
	if [  "$DATASET_TYPE" = "NYU_VAL"  ]; then
		model_path="../models/adenet_merge_nyu_kinect_tum/neair-adenet-final"
    fi
	
	# predict kinect -- adenet_astrous_os8_val
	python predict_sequences.py \
	--model_path $model_path \
	--image_dir $image_dir \
	--save_dir $pred_save_dir \
	--pred_depth_dir_name _adenet-finetune-pred \
	--figure_dir_name _adenet-finetune-figure \
	--rgb_extend_name $rgb_extend \
	--depth_extend_name _depth.png \
	--zfill_length 5 \
	--model_type adenet_astrous_os8_concat \
	--net_height 160 \
	--net_width 240 \
	--image_height 480 \
	--image_width 640 \
	--eval_dataset_type $DATASET_TYPE \
	--is_evaluate && 
	
	python evaluate_sequence.py \
	--depths_npy_dir $evaluate_depth_npy ;
	
fi

