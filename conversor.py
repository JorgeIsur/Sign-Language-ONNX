import tensorflow as tf
import tf2onnx


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('Tensorflow/workspace/models/señas_uam_0_1/', 'ckpt-21')).expect_partial()

# Carga el modelo desde un archivo de checkpoint
loaded_model = tf.keras.models.load_model("Tensorflow/workspace/models/señas_uam_0_1/.ckpt")

# Convierte el modelo a ONNX
onnx_model = tf2onnx.convert.from_keras(loaded_model)

# Guarda el modelo ONNX en un archivo
with open("modelo.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
