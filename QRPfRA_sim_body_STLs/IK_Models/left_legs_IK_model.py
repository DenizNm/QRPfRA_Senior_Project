import os
import warnings

import numpy as np
import pytest

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.error import Error
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env


class QRPfRA_v3(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, xml_file="/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA_sim_body_STLs/qrpfra_v3_leg_ik_scene.xml", frame_skip=1, **kwargs):
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

        obs_size = self.data.qpos.size + self.data.qvel.size

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def step(self, action):
        x_position_before = self.data.qpos[0]

        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]

        observation = self._get_obs()
        reward = x_position_after - x_position_before
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, False, False, info

    def _get_obs(self):
        #position = self.data.qpos.flat.copy()
        #velocity = self.data.qvel.flat.copy()
        sensor_data = self.data.sensordata.flat.copy()

        return sensor_data

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {"works": True}



env = QRPfRA_v3()
check_env(env)
obs = env.reset()
print(obs)


for i in range(-100, 100):
    for j in range(-100, 100):
        for k in range(-100, 100):

            env.render_mode = "human"
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step([float(m) for m in [i, j, k]])
            print(obs)
            if done:
                env.reset()

env.close()