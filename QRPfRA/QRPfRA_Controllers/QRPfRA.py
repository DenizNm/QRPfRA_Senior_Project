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
from time import sleep as sl

class QRPfRA_v3(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self,
                 xml_file="~/vscode_projects/QRPfRA_Senior_Project/QRPfRA/qrpfra_v3_scene.xml",
                 frame_skip=1, raspberry_pi=False, **kwargs):

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

        self.raspberry_pi = raspberry_pi
        
        if self.raspberry_pi:
            from adafruit_servokit import ServoKit
            self.kit = ServoKit(channels=16)
            for servo_num in range(0,12):
                self.kit.servo[servo_num].actuation_range = 180



    def step(self, action):
        if self.raspberry_pi:
            action_to_send = action * 90 + 90
            action_to_send = np.clip(action_to_send, 0, 180)
            action_to_send[1:3] = 180 - action_to_send[1:3]
            action_to_send[4:6] = 180 - action_to_send[4:6]

            action_to_send[0] = action_to_send[0] - 30
            action_to_send[1] = action_to_send[1] - 30
            action_to_send[2] = action_to_send[2] - 30
            
            action_to_send[3] = action_to_send[3] - 40
            action_to_send[4] = action_to_send[4] - 10
            action_to_send[5] = action_to_send[5] - 30
            
            action_to_send[6] = action_to_send[6]
            action_to_send[7] = action_to_send[7]
            
            action_to_send[8] = action_to_send[8]
            
            action_to_send[9] = action_to_send[9] + 10
            action_to_send[10] = action_to_send[10]
            action_to_send[11] = action_to_send[11]

            
            action_to_send = np.clip(np.array(action_to_send, dtype=np.uint8), 0, 180)
            print(action_to_send)

            for servo_num, angle in enumerate(action_to_send):
                self.kit.servo[servo_num].angle = angle



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


        self.step_count += 1

        reward = int(reward)
        return observation, reward, done, False, info # done, false, info

    def _get_obs(self):
        sensor_data = self.data.sensordata.flat.copy()
        return sensor_data


    def _unionize_action_buffer(self, action_buffer):
        """ each degree must consists of 3 integers totaling 36 integers, if an incoming angle for eaxmple 72,
         it must be converted to 072 rounded to 3 integers. """
        return [f"{angle:03d}" for angle in action_buffer]


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

        x, y, z = tuple(baselink_pos)

        reward = (y*10) ** 3 - abs(x) * 10

        if FL_hind_limb < 0 or RL_hind_limb < 0 or FR_hind_limb > 0 or RR_hind_limb > 0:
            reward -= 10000

        if observation[2] < -4 and self.step_count > 100:
            reward -= 50000
        return reward
