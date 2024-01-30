import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.error import Error
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env

import tensorflow as tf

right_leg_interpreter = tf.lite.Interpreter(model_path='/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/right_leg_model_quantized.tflite')
right_leg_interpreter.allocate_tensors()

left_leg_interpreter = tf.lite.Interpreter(model_path='/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/left_leg_model_quantized.tflite')
left_leg_interpreter.allocate_tensors()


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

        obs_size = len(self.data.sensordata.flat.copy())
        self.step_count = 0

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        self.left_leg_model = left_leg_interpreter
        self.right_leg_model = right_leg_interpreter


    def step(self, action):
        #Uncomment to run inference for TFlite inverse kinematics models
        action = np.clip(action, self.action_space.low, self.action_space.high).tolist()

        FL = self._run_inference(self.left_leg_model, action[0:3])
        RL = self._run_inference(self.left_leg_model, action[3:6])

        FR = self._run_inference(self.right_leg_model, action[6:9])
        RR = self._run_inference(self.right_leg_model, action[9:12])

        action = np.array([FL, RL, FR, RR]).flatten()

        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        #print("Obs:", observation)

        reward = self._compute_reward(observation, action) - 100

        info = {}

        if self.render_mode == "human":
            self.render()

        done = False

        if observation[2] < -8:
            reward -= 200

        self.step_count += 1
        if self.step_count > 20000:
            done = True

        #### LOOK AT HERE #### LOOK AT HERE #### LOOK AT HERE #### LOOK AT HERE
        #observation = observation/1000
        #### LOOK AT HERE #### LOOK AT HERE #### LOOK AT HERE #### LOOK AT HERE

        return observation, reward, done, info # done, false, info

    def _get_obs(self):
        sensor_data = self.data.sensordata.flat.copy()
        sensor_data[23:27] = [1 if i > 0 else 0 for i in sensor_data[23:27]]

        return sensor_data

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self.step_count = 0
        return observation

    def _get_reset_info(self):
        return {"works": True}

    def _run_inference(self, model, input_data):
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)

        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        output_data = model.get_tensor(output_details[0]['index'])
        return output_data

    def _compute_reward(self, observation, action):
        # Get absolute position of the base
        baselink_pos = self.get_body_com("base_link")
        x, y, z = tuple(baselink_pos)
        if y >= 0:
            reward = (y*20+2) ** 2 - abs(x) * 10
            if 4 < observation[1] and abs(observation[0]) < 4:
                reward += 8
            else:
                reward -= 8
        else:
            reward = (y) * 20 - abs(x) * 10


        if (0.2 < observation[21] or observation[21] == -1 and
                0.2 < observation[22] or observation[22] == -1):
            reward += 10
        else:
            reward -= 10

        return reward
