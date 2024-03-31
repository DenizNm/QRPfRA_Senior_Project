import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.error import Error
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env
import tensorflow as tf
import os
import serial
from scipy.spatial.transform import Rotation as R
import time


class QRPfRA_v3(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self,
                 xml_file="/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/qrpfra_v3_scene.xml",
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
        self.serial_port_name = "/dev/tty.usbserial-110"
        self.baud_rate = 115200

        if self.use_serial_port:
            self.serial_port = serial.Serial(self.serial_port_name, self.baud_rate)
            print("Serial port opened")
            time.sleep(2)

    def step(self, action):
        # Action is between -1 and 1
        action_np = np.array(action)
        if self.use_serial_port:
            self.send_over_serial(action_np)


        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()

        reward = self._compute_reward(observation, action) - 100
        info = {}

        if self.render_mode == "human":
            self.render()

        done = False

        if observation[2] < -8:
            reward -= 500
            done = True

        #### LOOK AT HERE #### LOOK AT HERE #### LOOK AT HERE #### LOOK AT HERE
        self.step_count += 1
        """if self.step_count > 10000:
            done = True"""

        reward = int(reward)
        #print("Reward:", reward)

        return observation, reward, done, False, info # done, false, info

    def _get_obs(self):
        sensor_data = self.data.sensordata.flat.copy()
        sensor_data[23:27] = [1.0 if i > 0.0 else 0.0 for i in sensor_data[23:27]]
        return sensor_data


    def _unionize_action_buffer(self, action_buffer):
        """ each degree must consists of 3 integers totaling 36 integers, if an incoming angle for eaxmple 72,
         it must be converted to 072 rounded to 3 integers. """
        return [f"{angle:03d}" for angle in action_buffer]

    def send_over_serial(self, action):
        #TODO: Convert the function to be compatible with Rasberry Pi
        action_to_send = action * 60 + 60
        action_to_send = np.clip(action_to_send, 0, 120)
        """action_to_send[1:3] = 120 - action_to_send[1:3]
        action_to_send[4:6] = 120 - action_to_send[4:6]"""
        #action_to_send[6:] = 120 - action_to_send[6:]
        # Convert to integer
        action_to_send = action_to_send.astype(int)
        print("Action to send:", action_to_send)
        print("Action to send shape:", action_to_send.shape)

        # Ensure data_array has the correct size and type
        if not isinstance(action_to_send, np.ndarray) or action_to_send.size != 12:
            raise ValueError("data_array must be a NumPy array of 12 integers.")

        # Convert each integer to a 3-character string, concatenate, and enclose in < and >
        data_str = '<' + ''.join([f"{num:03}" for num in action_to_send]) + '>'

        # Establish a serial connection
        self.serial_port.write(data_str.encode('utf-8') + b'\n')
        print("Data sent:", data_str.encode('utf-8'))
        #time.sleep(0.02)  # Wait for the Arduino to process the data
        response = self.serial_port.readline().decode('utf-8').strip()
        if response:  # Only print if there's a response
            print("Arduino response:", response)
        #time.sleep(0.5)  # Add a small delay to prevent spamming the Arduino

        """# print("Action:", action_to_send)

        action_to_send = ''.join(map(str, action_to_send))  # Convert list to string
        print("Action to send:", action_to_send)
        action_to_send = action_to_send.encode()  # Encode string to bytes
        self.serial_port.write(action_to_send)  # Write bytes to serial port
        # msg = self.serial_port.read(1)
        # print("Message:", msg.decode('utf-8'))"""


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

        reward = (y*10) ** 3 - abs(x) * 10



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
