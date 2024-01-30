import tensorflow as tf
from tensorflow.keras import layers

def get_critic(state_input_shape, action_input_shape):
    # Define the input layer with the new shape for state
    state_input = layers.Input(shape=state_input_shape)
    reshaped_states = layers.Reshape((state_input_shape[1]*state_input_shape[2], state_input_shape[3]))(state_input)

    # Add a bidirectional LSTM layer for state
    state_out = layers.Bidirectional(layers.LSTM(16, return_sequences=True))(reshaped_states)
    state_out = layers.Flatten()(state_out)
    state_out = layers.Dense(64, activation="relu")(state_out)  # Add this line

    # Action as input
    # Action as input
    action_input = layers.Input(shape=action_input_shape)
    """reshaped_action_input = layers.Reshape(
        (action_input_shape[1] * action_input_shape[2], action_input_shape[3]))(action_out)
    action_out = layers.Bidirectional(layers.LSTM(8, return_sequences=True))(reshaped_action_input)
    action_out = layers.Flatten()(action_out)
    action_out = layers.Dense(64, activation="relu")(action_out)"""
    action_out = layers.Flatten()(action_input)
    action_out = layers.Dense(64, activation="relu")(action_out)


    # Both are passed through separate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    # Add dense layers
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(128, activation="relu")(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(32, activation="relu")(out)

    # Define the output layer
    outputs = layers.Dense(1)(out)

    # Outputs single value for given state-action
    model = tf.keras.Model([state_input, action_input], outputs, name="Critic")

    return model