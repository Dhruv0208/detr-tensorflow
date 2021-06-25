# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import tensorflow as tf
import numpy as np
import cv2
import time

from detr_tf.training_config import TrainingConfig, training_config_parser

from detr_tf.networks.detr import get_detr_model

from detr_tf.data import processing
from detr_tf.data.coco import COCO_CLASS_NAME
from detr_tf.inference import get_model_inference, numpy_bbox_to_image


@tf.function
def run_inference(model, images, config, use_mask=True):

    if use_mask:
        mask = tf.zeros((1, images.shape[1], images.shape[2], 1))
        m_outputs = model((images, mask), training=False)
    else:
        m_outputs = model(images, training=False)

    predicted_bbox, predicted_labels, predicted_scores = get_model_inference(
        m_outputs, config.background_class, bbox_format="xy_center")
    return predicted_bbox, predicted_labels, predicted_scores


def main(model, use_mask=True, image_resize=None):

    image = cv2.imread("test.jpeg")
    print(image.shape)
    # Convert to RGB and process the input image
    if image_resize is not None:
        image = cv2.resize(image, (image_resize[1], image_resize[0]))
    model_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model_input = processing.normalized_images(model_input, config)

    # GPU warm up
    [run_inference(model, np.expand_dims(model_input, axis=0), config, use_mask=use_mask) for i in range(3)]

    # Timimg
    tic = time.time()
    predicted_bbox, predicted_labels, predicted_scores = run_inference(model, np.expand_dims(model_input, axis=0), config, use_mask=use_mask)
    toc = time.time()
    print(f"Inference latency: {(toc - tic)*1000} ms")

    image = image.astype(np.float32)
    image = image / 255
    image = numpy_bbox_to_image(image, predicted_bbox, labels=predicted_labels, scores=predicted_scores, class_name=COCO_CLASS_NAME)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = TrainingConfig()
    parser = training_config_parser()
    
    # Logging
    args = parser.parse_args()
    config.update_from_args(args)
    
    args.choosen_model == "detr"
    print("Loading detr...")
        # Load the model with the new layers to finetune
    model = get_detr_model(config, include_top=True, weights="detr")
    config.background_class = 91
    use_mask = True
    # Run webcam inference
    main(model, use_mask=use_mask, image_resize=args.image_resize)
