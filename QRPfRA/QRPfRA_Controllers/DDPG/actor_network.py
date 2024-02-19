import tensorflow as tf
from tensorflow.keras import layers

def get_actor(input_shape, num_actions):
    # Initialize weights between -0.5 and 0.5
    last_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)

    # Define the input layer with the new shape
    inputs = layers.Input(shape=input_shape)

    # Reshape the 3D input to 2D for LSTM
    reshaped = layers.Reshape((input_shape[1]*input_shape[2], input_shape[3]))(inputs)

    # Add a bidirectional LSTM layer
    out1 = layers.Bidirectional(layers.LSTM(16, return_sequences=True))(reshaped)
    out1 = layers.Flatten()(out1)

    # Add dense layers for out1
    out1 = layers.Dense(64, activation="relu", activity_regularizer=tf.keras.regularizers.L2(0.01))(out1)
    out1 = layers.Dense(256, activation="relu", activity_regularizer=tf.keras.regularizers.L2(0.01))(out1)
    out1 = layers.BatchNormalization()(out1)
    out1 = layers.Dense(128, activation="relu", activity_regularizer=tf.keras.regularizers.L2(0.01))(out1)
    out1 = layers.Dense(32, activation="linear", activity_regularizer=tf.keras.regularizers.L2(0.01))(out1)

    # Define the output layer with the custom initializer
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out1)

    # Create the model
    model = tf.keras.Model(inputs, outputs, name="Actor")
    return model