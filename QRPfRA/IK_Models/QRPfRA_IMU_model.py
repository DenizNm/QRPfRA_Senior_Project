import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.error import Error
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env
import tensorflow as tf
from QRPfRA_IMU import QRPfRA_v3

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.error import Error
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env
import tensorflow as tf
import os
import serial as pyserial
from scipy.spatial.transform import Rotation as R


class QRPfRA_v3(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self,
                 xml_file="/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/qrpfra_v3_scene_IMU.xml",
                 frame_skip=1, use_serial_port=False, **kwargs):

        utils.EzPickle.__init__(self, xml_file, frame_skip, **kwargs)
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config={},
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = len(self.data.sensordata.flat.copy())
        self.step_count = 0

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.use_serial_port = use_serial_port

        if self.use_serial_port:
            self.serial_port = pyserial.Serial('/dev/tty.usbserial-1110', 9600)
            if not self.serial_port.is_open:
                self.serial_port.open()

    def step(self, action):
        # Uncomment to run inference for TFlite inverse kinematics models

        # Action is between -1 and 1

        if self.use_serial_port:
            self.sent_over_serial(action)

        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        # print(observation)

        rotation = R.from_quat(observation[27:])
        # print(rotation)
        coordinates = "YZX"
        euler_angles = rotation.as_euler(coordinates, degrees=True)
        if euler_angles[2] <= 0:
            euler_angles[2] = -(euler_angles[2] + 180)
        else:
            euler_angles[2] = 180 - euler_angles[2]

        # print(f"Euler Angles ({coordinates}):", euler_angles)

        reward = self._compute_reward(observation, action) - 100
        info = {}

        if self.render_mode == "human":
            self.render()

        done = False

        reward = int(reward)
        # print("Reward:", reward)

        # obs = np.concatenate(np.array(observation[0:6]), np.array(euler_angles))
        # print("environment observation:", obs)

        return observation, euler_angles, reward, done, False, info  # done, false, info

    def _get_obs(self):
        sensor_data = self.data.sensordata.flat.copy()
        sensor_data[23:27] = [1.0 if i > 0.0 else 0.0 for i in sensor_data[23:27]]

        return sensor_data

    def _unionize_action_buffer(self, action_buffer):
        """ each degree must consists of 3 integers totaling 36 integers, if an incoming angle for eaxmple 72,
         it must be converted to 072 rounded to 3 integers. """
        return [f"{angle:03d}" for angle in action_buffer]

    def sent_over_serial(self, action):
        action_to_send = action * 60 + 60
        action_to_send = np.clip(action_to_send, 0, 120)
        action_to_send[1:3] = 120 - action_to_send[1:3]
        action_to_send[4:6] = 120 - action_to_send[4:6]
        action_to_send = action_to_send.astype(np.uint8)
        action_to_send = self._unionize_action_buffer(action_to_send)
        # print("Action:", action_to_send)

        action_to_send = ''.join(map(str, action_to_send))  # Convert list to string
        print("Action to send:", action_to_send)
        action_to_send = action_to_send.encode()  # Encode string to bytes
        self.serial_port.write(action_to_send)  # Write bytes to serial port
        # msg = self.serial_port.read(1)
        # print("Message:", msg.decode('utf-8'))

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self.step_count = 0
        return observation

    def _get_reset_info(self):
        return {"works": True}

    def _compute_reward(self, observation, action):
        # Get absolute position of the base
        baselink_pos = self.get_body_com("base_link")
        FL_hind_limb = observation[9]
        RL_hind_limb = observation[12]
        FR_hind_limb = observation[15]
        RR_hind_limb = observation[18]
        foot_observation = sum(observation[23:27])

        x, y, z = tuple(baselink_pos)

        reward = (y * 10) ** 3 - abs(x) * 10

        if (0.2 < observation[21] or observation[21] == -1 and
                0.2 < observation[22] or observation[22] == -1):
            reward += 10
        else:
            reward -= 10

        if FL_hind_limb < 0 or RL_hind_limb < 0 or FR_hind_limb > 0 or RR_hind_limb > 0:
            reward -= 10000

        if foot_observation >= 3:
            reward += foot_observation * 50
        else:
            reward -= 1500

        if observation[2] < -4 and self.step_count > 100:
            reward -= 50000

        return reward


env = QRPfRA_v3()
obs = env.reset_model()
print(obs)


right_leg_model = tf.lite.Interpreter(model_path='/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/left_leg_model_quantized.tflite')
right_leg_model.allocate_tensors()

left_leg_model = tf.lite.Interpreter(model_path='/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/left_leg_model_quantized.tflite')
left_leg_model.allocate_tensors()


def init_position():
    front_left_leg = np.array([-0.05, 0.0, -0.1])
    back_left_leg = np.array([-0.05, 0.0, -0.105])

    front_right_leg = np.array([-0.05, 0.0, -0.1])
    back_right_leg = np.array([-0.05, 0.0, -0.105])
    return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))


def second_pose():
    front_left_leg = np.array([-0.15, 0, -0.12])
    back_left_leg = np.array([-0.15, 0, -0.125])

    front_right_leg = np.array([0.15, 0, -0.12])
    back_right_leg = np.array([0.15, 0, -0.125])
    return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))


def third_pose():
    front_left_leg = np.array([-0.05, -0.12, -0.15])
    back_left_leg = np.array([-0.05, -0.12, -0.155])

    front_right_leg = np.array([-0.05, -0.12, -0.15])
    back_right_leg = np.array([-0.05, -0.12, -0.155])
    return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))


def generate_points(input_list, total_step):
    step_between_points = total_step // (len(input_list) - 1)
    output = []
    for i in range(len(input_list) - 1):
        output.extend(np.linspace(input_list[i], input_list[i + 1], step_between_points))
    return output


def trot_generation(steps=100, reverse=False, fr_height_diff=0.0):
    x_keyframes_d1 = [-0.05, -0.05, -0.05, -0.05]
    y_keyframes_d1 = [0.0, -0.08, 0.0, 0.08]
    z_keyframes_d1 = [-0.12, -0.13, -0.1, -0.13]

    x_keyframes_d2 = [-0.05, -0.05, -0.05, -0.05]
    y_keyframes_d2 = [0.0, 0.08, 0.0, -0.08]
    z_keyframes_d2 = [-0.1, -0.13, -0.13, -0.12]


    # Interpolate as much as the number of steps
    x_d1 = generate_points(x_keyframes_d1, steps+1)
    y_d1 = generate_points(y_keyframes_d1, steps+1)
    z_d1 = generate_points(z_keyframes_d1, steps+1)

    x_d2 = generate_points(x_keyframes_d2, steps+1)
    y_d2 = generate_points(y_keyframes_d2, steps+1)
    z_d2 = generate_points(z_keyframes_d2, steps+1)

    front_left_rear_right = np.array([x_d1, y_d1, z_d1])
    front_right_rear_left = np.array([x_d2, y_d2, z_d2])

    if not reverse:
        trot_action = np.concatenate(
            (front_left_rear_right, front_right_rear_left, front_right_rear_left, front_left_rear_right)).T
    elif reverse:
        trot_action = np.concatenate(
            (front_left_rear_right, front_right_rear_left, front_right_rear_left, front_left_rear_right)).T

    trot_action[:, 5] = trot_action[:, 5] + fr_height_diff
    trot_action[:, 11] = trot_action[:, 11] + fr_height_diff
    return trot_action


def _run_inference(model, input_data):
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)

    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data

def phase_transition_controller(act_in, act_out):
    err = act_out - act_in
    action = act_in + 0.08 * err
    return action


def action_xyz_to_angle(action_xyz, current_action_angle):
    if type(action_xyz) != list:
        action_xyz = action_xyz.tolist()
    elif type(action_xyz) == list:
        pass
    else:
        raise ValueError("Invalid data type for action_xyz")

    FL = _run_inference(left_leg_model, action_xyz[0:3])
    RL = _run_inference(left_leg_model, action_xyz[3:6])

    FR = _run_inference(right_leg_model, action_xyz[6:9])
    RR = _run_inference(right_leg_model, action_xyz[9:12])

    action_angle = np.array([FL, RL, FR, RR]).flatten()

    action_angle = phase_transition_controller(current_action_angle, action_angle)
    return action_angle


IMU_set = [] # [obs0, obs1, obs2, obs3, obs4, obs5, act0, act1, act2]

current_action_angle = np.zeros(12).flatten()
step = 0
# env.render_mode = "human"

total_steps = 75
generated_action = trot_generation(total_steps, reverse=False, fr_height_diff=-0.02)
print("Shape is:", generated_action.shape)

in_step_cnt = 0
while True:
    if step > 100:
        action_xyz = generated_action[in_step_cnt, :]
        if in_step_cnt == (total_steps - 2):
            in_step_cnt = 0

        in_step_cnt += 1

    elif step <= 100:
        action_xyz = second_pose()

    action_angle = action_xyz_to_angle(action_xyz, current_action_angle)
    obs, orientation, reward, done, _, info = env.step(action_angle)
    current_action_angle = action_angle

    obs = obs[0:6]
    obs = list(obs) + list(orientation)

    IMU_set.append(obs)

    if step >= 10000:
        break

    print(step)

    if done:
        env.reset()

    step += 1


print(len(IMU_set))
IMU_set = IMU_set[1:]
print(len(IMU_set))


def generate_dataset_for_LSTM_model(data_x, data_y, time_steps):
    X, y = [], []
    for i in range(len(data_x) - time_steps):
        v = data_x[i:i + time_steps]
        X.append(v)
        y.append(data_y[i + time_steps])
    return np.array(X), np.array(y)

#Convert left leg list to numpy array
IMU_set_as_np = np.array(IMU_set)


print(IMU_set_as_np[0:9])
print("*"*50)
observations = (IMU_set_as_np[:, 0:6][:])/100
print(observations)
print("*"*50)
actions = (IMU_set_as_np[:, 6:][:])/180
print(actions)

LSTM_time_steps = 10
observations, actions = generate_dataset_for_LSTM_model(observations, actions, LSTM_time_steps)



import tensorflow as tf
from sklearn.model_selection import train_test_split
import datetime
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense



# Split the dataset into training and testing sets
train_obs, test_obs, train_actions, test_actions = train_test_split(observations, actions, test_size=0.1, shuffle=True)

# Define the model architecture

ik_input = layers.Input(shape=(LSTM_time_steps, 6))
x = Bidirectional(layers.LSTM(128, return_sequences=True))(ik_input)
x = Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = Bidirectional(layers.LSTM(32, return_sequences=True))(x)
x = layers.LSTM(16)(x)
x = layers.Dense(256, activation='relu')(x)
x1 = layers.Dense(1, activation='tanh')(x)
x2 = layers.Dense(1, activation='tanh')(x)
x3 = layers.Dense(1, activation='tanh')(x)
ik_output = layers.concatenate([x1, x2, x3])


model = models.Model(ik_input, ik_output)
# Compile the model
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)  # Set the learning rate here
model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['accuracy'])

# Train the model
model.fit(observations, actions, epochs=10, batch_size=64)

# Evaluate the model on the testing set
test_loss = model.evaluate(test_obs, test_actions)
print('Test Loss:', test_loss)
current_time = datetime.datetime.now()
print("Current Time:", current_time)


print(test_obs[0:1])
pred = model.predict(test_obs[0:1])
print("Prediction ", pred)

print("Reality ", test_actions[0:1])
