import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


SAVE_VIDEO = True
RUN_FIDUCIAL_DETECTION = True
RUN_OBJECT_DETECTION = True

# Load image paths
#VIDEO_PATH = "../../../Data/scenario_1/MOV_0012.MP4"
VIDEO_PATH = "../../../Data/scenario_1/MOV_0022.MP4"
frame = 3200
#VIDEO_PATH = "../../../Data/scenario_2/MOV_0020.MP4"
#frame = 3800

# Path to trained model
#PATH_TO_MODEL_DIR = "exported-models/my_ssd_resnet50_v1_fpn/"
PATH_TO_MODEL_DIR = "exported-models/my_efficientdet_d1/"

# Label
PATH_TO_LABELS = "annotations/label_map.pbtxt"

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

if RUN_OBJECT_DETECTION:
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
import apriltag

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)

# define 
if SAVE_VIDEO:
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))



count = frame
while(cap.isOpened()):
	print('Running inference for {}... '.format(VIDEO_PATH), end='')

	ret, frame = cap.read()
	image_np_with_detections = frame.copy()
	print("frame: ", count)
	
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if RUN_FIDUCIAL_DETECTION:
		detector = apriltag.Detector()
		detections = detector.detect(gray_frame)
		
		NR_DETECTIONS = len(detections)
		print("# detected fiducials", NR_DETECTIONS)
		for i in range(NR_DETECTIONS):
			# Get and draw pose of fiducial
			# fx, fy, cx, cy
			K = [929.7, 932.749, 616.4179214005839, 335.4107412307333]
			pose, _, _ = detector.detection_pose(detections[0], K , tag_size=1, z_sign=1 )
			
			apriltag._draw_pose(image_np_with_detections, K, 1, pose, z_sign=1)
		# We want to draw four lines
			#for j in range(4):
				#start_point = tuple(detections[i].corners[j-1, :].astype(int))
				#end_point = tuple(detections[i].corners[j, :].astype(int))
				#cv2.line(image_np_with_detections, start_point,
					#end_point, (255, 0, 0),
					#thickness=3)

	if RUN_OBJECT_DETECTION:
		# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
		input_tensor = tf.convert_to_tensor(frame)
		# The model expects a batch of images, so add an axis with `tf.newaxis`.
		input_tensor = input_tensor[tf.newaxis, ...]
		# input_tensor = np.expand_dims(image_np, 0)
		detections = detect_fn(input_tensor)

		# All outputs are batches tensors.
		# Convert to numpy arrays, and take index [0] to remove the batch dimension.
		# We're only interested in the first num_detections.
		num_detections = int(detections.pop('num_detections'))
		detections = {key: value[0, :num_detections].numpy()
		               for key, value in detections.items()}
		detections['num_detections'] = num_detections

		# detection_classes should be ints.
		detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

		category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
	                                                                    use_display_name=True)

		viz_utils.visualize_boxes_and_labels_on_image_array(
			image_np_with_detections,
			detections['detection_boxes'],
			detections['detection_classes'],
			detections['detection_scores'],
			category_index,
			use_normalized_coordinates=True,
			max_boxes_to_draw=10,
			min_score_thresh=.60,
			agnostic_mode=False)

	cv2.imshow('frame',image_np_with_detections)
	if cv2.waitKey(2) & 0xFF == ord('q'):
		break

	count = count + 1
	if SAVE_VIDEO:
		out.write(image_np_with_detections)




cap.release()
cv2.destroyAllWindows()