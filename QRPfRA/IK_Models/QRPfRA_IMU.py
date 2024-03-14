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
        #Uncomment to run inference for TFlite inverse kinematics models


        # Action is between -1 and 1

        if self.use_serial_port:
            self.sent_over_serial(action)


        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        print(observation)

        """rotation = R.from_quat(observation[27:])
        print(rotation)
        coordinates = "YZX"
        euler_angles = rotation.as_euler(coordinates, degrees=True)
        if euler_angles[2] <= 0:
            euler_angles[2] = -(euler_angles[2] + 180)
        else:
            euler_angles[2] = 180 - euler_angles[2]"""
        
        

        print(f"Euler Angles ({coordinates}):", euler_angles)

        reward = self._compute_reward(observation, action) - 100
        info = {}

        if self.render_mode == "human":
            self.render()

        done = False

        reward = int(reward)
        #print("Reward:", reward)
        
        #obs = np.concatenate(np.array(observation[0:6]), np.array(euler_angles))
        #print("environment observation:", obs)

        return observation, reward, done, False, info # done, false, info

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
