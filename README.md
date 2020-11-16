# asset_tracking

calibrate.py: ./calibrate.py ../../../Data/calibration/MOV_0007_calibration.mp4 MOV_0007.yaml --debug-dir out -fs 20 --debug-dir ~/Downloads/debug/


generate_tfrecord.py: python generate_tfrecord.py -x ../workspace/container_detection/images/test -l ../workspace/container_detection/annotations/label_map.pbtxt -o ../workspace/container_detection/annotations/test.record

python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config --checkpoint_dir=models/my_ssd_resnet50_v1_fpn


run eval on exported model

python model_main_tf2.py --model_dir=exported-models/my_efficientdet_d1_cutout_rscps_contrast --pipeline_config_path=exported-models/my_efficientdet_d1_cutout_rscps_contrast/pipeline.config --checkpoint_dir=exported-models/my_efficientdet_d1_cutout_rscps_contrast/checkpoint
