import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.error import Error
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env

import tensorflow as tf
from tensorflow import keras

class QRPfRA_v3(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self,
                 xml_file="/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA_sim_body_STLs/qrpfra_v3_scene.xml",
                 frame_skip=1, **kwargs):

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
        self.left_leg_model = tf.keras.models.load_model("/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA_sim_body_STLs/IK_Models/right_legs_model")
        self.right_leg_model = tf.keras.models.load_model("/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA_sim_body_STLs/IK_Models/left_legs_model")

    def step(self, action):
        print("Position of the foothold", action)
        #x_position_before = self.data.qpos[0]
        action = np.array(action).reshape(1, 3)
        action = self.left_leg_model.predict(action)[0]*100

        print("Angle action", action)

        self.do_simulation(action, self.frame_skip)
        #x_position_after = self.data.qpos[0]

        observation = self._get_obs()
        reward = 0  #x_position_after - x_position_before
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
env.render_mode = "human"