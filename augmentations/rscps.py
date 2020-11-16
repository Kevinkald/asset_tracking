from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import os
import xml.etree.ElementTree as ET



import functools
import inspect
import sys

import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from tensorflow.python.ops import control_flow_ops
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import densepose_ops
from object_detection.core import keypoint_ops
from object_detection.core import preprocessor_cache
from object_detection.core import standard_fields as fields
from object_detection.utils import autoaugment_utils
from object_detection.utils import ops
from object_detection.utils import patch_ops
from object_detection.utils import shape_utils

def random_scale_crop_and_pad_to_square(
    image,
    label_weights=None,
    masks=None,
    boxes=None,
    keypoints=None,
    label_confidences=None,
    scale_min=0.1,
    scale_max=2.0,
    output_size=512,
    resize_method=tf.image.ResizeMethod.BILINEAR,
    seed=None):
  """Randomly scale, crop, and then pad an image to fixed square dimensions.

   Randomly scale, crop, and then pad an image to the desired square output
   dimensions. Specifically, this method first samples a random_scale factor
   from a uniform distribution between scale_min and scale_max, and then resizes
   the image such that it's maximum dimension is (output_size * random_scale).
   Secondly, a square output_size crop is extracted from the resized image
   (note, this will only occur when random_scale > 1.0). Lastly, the cropped
   region is padded to the desired square output_size, by filling with zeros.
   The augmentation is borrowed from [1]
   [1]: https://arxiv.org/abs/1911.09070

  Args:
    image: rank 3 float32 tensor containing 1 image ->
      [height, width, channels].
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4]. Boxes
      are in normalized form meaning their coordinates vary between [0, 1]. Each
      row is in the form of [ymin, xmin, ymax, xmax]. Boxes on the crop boundary
      are clipped to the boundary and boxes falling outside the crop are
      ignored.
    labels: rank 1 int32 tensor containing the object classes.
    label_weights: float32 tensor of shape [num_instances] representing the
      weight for each box.
    masks: (optional) rank 3 float32 tensor with shape [num_instances, height,
      width] containing instance masks. The masks are of the same height, width
      as the input `image`.
    keypoints: (optional) rank 3 float32 tensor with shape [num_instances,
      num_keypoints, 2]. The keypoints are in y-x normalized coordinates.
    label_confidences: (optional) float32 tensor of shape [num_instance]
      representing the confidence for each box.
    scale_min: float, the minimum value for the random scale factor.
    scale_max: float, the maximum value for the random scale factor.
    output_size: int, the desired (square) output image size.
    resize_method: tf.image.ResizeMethod, resize method to use when scaling the
      input images.
    seed: random seed.

  Returns:
    image: image which is the same rank as input image.
    boxes: boxes which is the same rank as input boxes.
           Boxes are in normalized form.
    labels: new labels.
    label_weights: rank 1 float32 tensor with shape [num_instances].
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
    label_confidences: confidences for retained boxes.
  """
  img_shape = tf.shape(image)
  input_height, input_width = img_shape[0], img_shape[1]
  random_scale = tf.random_uniform([], scale_min, scale_max, seed=seed)

  # Compute the scaled height and width from the random scale.
  max_input_dim = tf.cast(tf.maximum(input_height, input_width), tf.float32)
  input_ar_y = tf.cast(input_height, tf.float32) / max_input_dim
  input_ar_x = tf.cast(input_width, tf.float32) / max_input_dim
  scaled_height = tf.cast(random_scale * output_size * input_ar_y, tf.int32)
  scaled_width = tf.cast(random_scale * output_size * input_ar_x, tf.int32)

  # Compute the offsets:
  offset_y = tf.cast(scaled_height - output_size, tf.float32)
  offset_x = tf.cast(scaled_width - output_size, tf.float32)
  offset_y = tf.maximum(0.0, offset_y) * tf.random_uniform([], 0, 1, seed=seed)
  offset_x = tf.maximum(0.0, offset_x) * tf.random_uniform([], 0, 1, seed=seed)
  offset_y = tf.cast(offset_y, tf.int32)
  offset_x = tf.cast(offset_x, tf.int32)

  # Scale, crop, and pad the input image.
  scaled_image = tf.image.resize_images(
      image, [scaled_height, scaled_width], method=resize_method)
  scaled_image = scaled_image[offset_y:offset_y + output_size,
                              offset_x:offset_x + output_size, :]
  output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, output_size,
                                              output_size)

  # Update the boxes.
  new_window = tf.cast(
      tf.stack([offset_y, offset_x,
                offset_y + output_size, offset_x + output_size]),
      dtype=tf.float32)
  new_window /= tf.cast(
      tf.stack([scaled_height, scaled_width, scaled_height, scaled_width]),
      dtype=tf.float32)
  #boxlist = box_list.BoxList(boxes)
  #boxlist = box_list_ops.change_coordinate_frame(boxlist, new_window)
  #boxlist, indices = box_list_ops.prune_completely_outside_window(
  #    boxlist, [0.0, 0.0, 1.0, 1.0])
  #boxlist = box_list_ops.clip_to_window(
  #    boxlist, [0.0, 0.0, 1.0, 1.0], filter_nonoverlapping=False)

  return_values = output_image

  if masks is not None:
    new_masks = tf.expand_dims(masks, -1)
    new_masks = tf.image.resize_images(
        new_masks, [scaled_height, scaled_width], method=resize_method)
    new_masks = new_masks[:, offset_y:offset_y + output_size,
                          offset_x:offset_x + output_size, :]
    new_masks = tf.image.pad_to_bounding_box(
        new_masks, 0, 0, output_size, output_size)
    new_masks = tf.squeeze(new_masks, [-1])
    return_values.append(tf.gather(new_masks, indices))

  if keypoints is not None:
    keypoints = tf.gather(keypoints, indices)
    keypoints = keypoint_ops.change_coordinate_frame(keypoints, new_window)
    keypoints = keypoint_ops.prune_outside_window(
        keypoints, [0.0, 0.0, 1.0, 1.0])
    return_values.append(keypoints)

  if label_confidences is not None:
    return_values.append(tf.gather(label_confidences, indices))

  return return_values

folder="../images/train_copy/"

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    # Accessing each jpg image
    if img is not None:
        # Create image augmentation

        filename_split = filename.split(".jpg")
        xml_filename_old = filename_split[0] + ".xml"

        new_filename = filename_split[0] + "_cutout.jpg"
        xml_filename = filename_split[0] + "_cutout.xml"

        # copy and edit old xml file
        tree = ET.parse(folder + xml_filename_old)
        root = tree.getroot()
        k = root.find('filename')
        k.text = new_filename
        l = root.find('path')
        l.text = new_filename

        image_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        # augment image
        image_aug = random_scale_crop_and_pad_to_square(image=image_tensor, output_size=640)
        #cv2.imshow("image", image_aug)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

   
        image_aug = np.float32(image_aug)
    	# save augmented image + edited xml file
        cv2.imwrite(new_filename, image_aug)
        tree.write(xml_filename)