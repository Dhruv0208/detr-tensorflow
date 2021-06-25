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
import os
import onnx
import tf2onnx
from pathlib import Path

from detr_tf.training_config import TrainingConfig, training_config_parser

from detr_tf.networks.detr import get_detr_model
# -

config = TrainingConfig()
model = get_detr_model(config, include_top = True, weights = "detr")

import tensorflow as tf

# +
import keras
import tensorflow.compat.v1.keras.backend as K
K.set_learning_phase(0)

def keras_to_pb(model, output_filename, output_node_names):
    in_name = model.layers[0].get_output_at(0).name.split(':')[0]

    if output_node_names is None:
        output_node_names = [model.layers[-1].get_output_at(0).name.split(':')[0]]

    sess = K.get_session()

   # The TensorFlow freeze_graph expects a comma-separated string of output node names.
    output_node_names_tf = ','.join(output_node_names)

    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
       sess,
       sess.graph_def,
       output_node_names)

    sess.close()
    wkdir = ''
    tf.train.write_graph(frozen_graph_def, wkdir, output_filename, as_text=False)

    return in_name, output_node_names


in_tensor_name, out_tensor_names = keras_to_pb(model, "models/resnet50.pb", None) 


# +
def get_model(config, args):
    print("Loading detr...")
    # Load the model with the new layers to finetune
    model = get_detr_model(config, include_top=True, weights="detr")
    config.background_class = 91
    m_output_names = ["pred_logits", "pred_boxes"]
    use_mask = True
    # model.summary()
    # return model


    input_image = tf.keras.Input(args.input_shape, batch_size=1, name="input_image")
    if use_mask:
        mask =tf.keras.Input(args.input_shape[:2] + [1], batch_size=1, name="input_mask")
        m_inputs = (input_image, mask)
    else:
        m_inputs = (input_image, )
    all_outputs = model(m_inputs, training=False)
    
    m_outputs = {
        name:tf.identity(all_outputs[name], name=name) 
        for name in m_output_names if name in all_outputs}
    [print(m_outputs[name]) for name in m_outputs]

    model =  tf.keras.Model(m_inputs, m_outputs, name=args.choosen_model)
    model.summary() 
    return model


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = TrainingConfig()
    parser = training_config_parser()

    args = parser.parse_args()
    config.update_from_args(args)

    args.input_shape.append(3) # C = 3
    args.choosen_model == "detr"
    if args.save_to is None:
        args.save_to = os.path.join("weights", args.choosen_model, args.choosen_model + "_trt")

    # === Load model
    model = get_model(config, args)
    # === Save model to pb file
    if not os.path.isdir(args.save_to):
        os.makedirs(args.save_to)

    # === Save onnx file
    input_spec = [tf.TensorSpec.from_tensor(tensor) for tensor in model.input]
    # print(input_spec)
    output_path = os.path.join(args.save_to, args.choosen_model + ".onnx")
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_spec, 
        opset=13, output_path=output_path)
    print("===== Inputs =======")
    [print(n.name) for n in model_proto.graph.input]
    print("===== Outputs =======")
    [print(n.name) for n in model_proto.graph.output]


