import tensorflow as tf

# Convert the TensorFlow models to TFLite models and quantize them
converter = tf.lite.TFLiteConverter.from_saved_model("/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/right_legs_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
right_leg_tflite_model = converter.convert()
with open('/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/right_leg_model_quantized.tflite', 'wb') as f:
    f.write(right_leg_tflite_model)


converter = tf.lite.TFLiteConverter.from_saved_model("/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/left_legs_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
left_leg_tflite_model = converter.convert()
with open('/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/right_leg_model_quantized.tflite', 'wb') as f:
    f.write(left_leg_tflite_model)

converter = tf.lite.TFLiteConverter.from_saved_model("/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/fine_tuned_legs_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
left_leg_tflite_model = converter.convert()
with open('/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/fine_tuned_leg_model_quantized.tflite', 'wb') as f:
    f.write(left_leg_tflite_model)


converter = tf.lite.TFLiteConverter.from_saved_model("/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/fine_tuned_legs_model_v2")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
left_leg_tflite_model = converter.convert()
with open('/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/fine_tuned_leg_model_quantized_v2.tflite', 'wb') as f:
    f.write(left_leg_tflite_model)





"""# Load the TFLite models for inference
right_leg_interpreter = tf.lite.Interpreter(model_path='right_leg_model.tflite')
right_leg_interpreter.allocate_tensors()

left_leg_interpreter = tf.lite.Interpreter(model_path='left_leg_model.tflite')
left_leg_interpreter.allocate_tensors()


# Run inference on the TFLite models
# Assuming `input_data` is your input
right_leg_interpreter.set_tensor(right_leg_interpreter.get_input_details()[0]['index'], input_data)
right_leg_interpreter.invoke()
right_leg_output = right_leg_interpreter.get_tensor(right_leg_interpreter.get_output_details()[0]['index'])

left_leg_interpreter.set_tensor(left_leg_interpreter.get_input_details()[0]['index'], input_data)
left_leg_interpreter.invoke()
left_leg_output = left_leg_interpreter.get_tensor(left_leg_interpreter.get_output_details()[0]['index'])
"""