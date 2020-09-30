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
IMAGE_PATHS = []
images = os.fsencode('images/test/')
for file in os.listdir(images):
	filename = os.fsdecode(file)
	if filename.endswith(".jpg"):
		IMAGE_PATHS.append('images/test/'+filename)

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

count = 0
for image_path in IMAGE_PATHS:
	print('Running inference for {}... '.format(image_path), end='')

	image_np = load_image_into_numpy_array(image_path)

	# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
	input_tensor = tf.convert_to_tensor(image_np)
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

	image_np_with_detections = image_np.copy()




	category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

	viz_utils.visualize_boxes_and_labels_on_image_array(
		image_np_with_detections,
		detections['detection_boxes'],
		detections['detection_classes'],
		detections['detection_scores'],
		category_index,
		use_normalized_coordinates=True,
		max_boxes_to_draw=200,
		min_score_thresh=.50,
		agnostic_mode=False)
	gr_im= Image.fromarray(image_np_with_detections).save('detections/' + str(count) + '.jpg')
	plt.figure()
	plt.imshow(image_np_with_detections)
	plt.show()
	print('Done')
	count = count + 1
plt.show()