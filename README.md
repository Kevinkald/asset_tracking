# asset_tracking

## Partition data set: 

This script randomly divides the labeled data(.jpg + corresponding .xml) in to 90% train and 10% eval data. 0.1 can be adjusted accordingly to different threshold. 

```python partition_dataset.py -x -i [PATH_TO_IMAGES_FOLDER] -o [PATH_TO_PARTITIONED_IMAGE_FOLDER] -r 0.1```


## Calibrating camera using video:


```calibrate.py: ./calibrate.py [PATH_TO_VIDEO] [OUTPUT_FILE_NAME].yaml --debug-dir out -fs 20 --debug-dir ~/Downloads/debug/```


## Generate tfrecord:

This converts .xml to .record format which is required for using TensorFlow 2 object detection API. Needs to be done for both training- and evaluation data set.

```generate_tfrecord.py: python generate_tfrecord.py -x [PATH_TO_TRAIN_OR_EVAL_IMAGES] -l [PATH_TO_LABEL_MAP]/label_map.pbtxt -o [PATH_TO_OUTPUT_TFRECORD_FILE]/test.record```


## Train a model:

```python train_model.py --model_dir=[PATH_TO_PRETRAINED_MODEL_TO_TRAIN] --pipeline_config_path=[PATH_TO_PIPELINE_CONFIG]pipeline.config```


## Run evaluation on a model:

Add the --checkpoint_dir attribute. NB!: set export CUDA_VISIBLE_DEVICES='' in the validation terminal to not interfere with the GPU.

```python train_model.py --model_dir=[PATH_TO_PRETRAINED_MODEL_TO_TRAIN] --pipeline_config_path=[PATH_TO_PIPELINE_CONFIG]pipeline.config --checkpoint_dir=[PATH_TO_CHECKPOINT_DIR]```

## Run evalulation on an exported model

python train_model.py --model_dir=[PATH_TO_EXPORTED_MODEL] --pipeline_config_path=[PATH_TO_PIPELINE_CONFIG]/pipeline.config --checkpoint_dir=[PATH_TO_MODEL_CHECKPOINT]/checkpoint

## Monitor evaluation using tensorboard

```tensorboard --logdir=[PATH_TO_TRAINING_MODEL_DIR]```
