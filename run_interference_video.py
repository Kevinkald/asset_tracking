import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load image paths
VIDEO_PATH = "../../../Data/scenario_1/MOV_0012.MP4"

# Path to trained model
PATH_TO_MODEL_DIR = "exported-models/my_ssd_resnet50_v1_fpn/"

# Label
PATH_TO_LABELS = "annotations/label_map.pbtxt"

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

print('Loading model...', end='')
start_time = time.time()
# Load saved model and build the detection function
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "saved_model"
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import PIL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
	return np.array(Image.open(path))

import cv2

cap = cv2.VideoCapture(VIDEO_PATH)

from cv2 import aruco

import apriltag

def draw(img, tags):
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    offset = 75

    for tag in tags:
        print(f">> {tag.tag_id} {tag.tag_family}")
        for idx in range(len(tag.corners)):
            cv2.line(color_img, 
                     tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), 
                     (0, 255, 0),
                     thickness=4)

        cv2.putText(color_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int),tag.corners[0, 1].astype(int)+offset),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    thickness = 6,
                    color=(0, 0, 255))
    return color_img


count = 0
while(cap.isOpened()):
	print('Running inference for {}... '.format(VIDEO_PATH), end='')

	ret, frame = cap.read()

	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	detector = apriltag.Detector()
	result = detector.detect(gray)

	cv2.imshow('frame',frame_markers)

	#frame_np = load_image_into_numpy_array(np.array(frame))

	# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
	# input_tensor = tf.convert_to_tensor(frame)
	# # The model expects a batch of images, so add an axis with `tf.newaxis`.
	# input_tensor = input_tensor[tf.newaxis, ...]
	# # input_tensor = np.expand_dims(image_np, 0)
	# detections = detect_fn(input_tensor)

	# # All outputs are batches tensors.
	# # Convert to numpy arrays, and take index [0] to remove the batch dimension.
	# # We're only interested in the first num_detections.
	# num_detections = int(detections.pop('num_detections'))
	# detections = {key: value[0, :num_detections].numpy()
	#                for key, value in detections.items()}
	# detections['num_detections'] = num_detections

	# # detection_classes should be ints.
	# detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

	# image_np_with_detections = frame.copy()

	# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
 #                                                                    use_display_name=True)

	# viz_utils.visualize_boxes_and_labels_on_image_array(
	# 	image_np_with_detections,
	# 	detections['detection_boxes'],
	# 	detections['detection_classes'],
	# 	detections['detection_scores'],
	# 	category_index,
	# 	use_normalized_coordinates=True,
	# 	max_boxes_to_draw=10,
	# 	min_score_thresh=.30,
	# 	agnostic_mode=False)

	# cv2.imshow('frame',image_np_with_detections)
	if cv2.waitKey(2) & 0xFF == ord('q'):
		break



cap.release()
cv2.destroyAllWindows()