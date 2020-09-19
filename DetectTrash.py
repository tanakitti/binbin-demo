import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import pathlib
import numpy as np
from PIL import Image
import scipy.misc
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import warnings

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def download_images(root_path):
    
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_path) for f in filenames if os.path.splitext(f)[1] == '.jpg']
    image_paths = []
    for filename in result:
        image_path = tf.keras.utils.get_file(fname=filename,
                                            origin=filename,
                                            untar=False)
        image_path = pathlib.Path(image_path)
        image_paths.append(str(image_path))
    return image_paths


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

# TEST_IMAGE_ROOT_PATH = "F:/_Archieve/FinalSeniorProject/ModelEval/Images"
LABEL_FILENAME = './Resource/annotations/label_map.pbtxt'
PATH_TO_SAVED_MODEL = "./Resource/my_model/saved_model"
# OUTPUT_EVAL_PATH = "F:/Binbin/workspace/training_demo/imagesEval/"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
detect_fn = model.signatures['serving_default']

end_time = time.time()
elapsed_time = end_time - start_time

print('Done! Took {} seconds'.format(elapsed_time))


category_index = label_map_util.create_category_index_from_labelmap(LABEL_FILENAME,
                                                                    use_display_name=True)
print(category_index)

def detectObjectFromImage(image_np):
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

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
    detections['detection_scores'] = detections['detection_scores'].astype(np.float64)

    print(detections)
    print(category_index)
    print(detections['detection_classes'])
    print(detections['detection_scores'])
    image_np_with_detections = image_np.copy()
        
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=5,
          min_score_thresh=.95,
          agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.savefig("output.jpg")

    return category_index[detections['detection_classes'][0]]['name'], detections['detection_scores'][0]

# path = "f:/_Archieve/FinalSeniorProject/ModelEval/Images/Plastic/test37.jpg"
# detectObjectFromImage(np.array(Image.open(path)))