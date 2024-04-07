import numpy as np
from gymnasium.utils.env_checker import check_env
import QRPfRA
from TROT import TROT
import os

env = QRPfRA.QRPfRA_v3(use_serial_port=False)
check_env(env)
obs = env.reset()
env.render_mode = "human"

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

controller = TROT(tf_lite_model_path=os.path.dirname(os.getcwd()).replace("\\", "/") + "/IK_Models/left_leg_model_quantized.tflite",
                  env_num_actions=num_actions,
                  env_num_states=num_states)

step = 0
current_obs = np.zeros(num_states)
current_action = np.zeros(num_actions)
current_angle = 0
while True:
    desired = 0
    action = controller.get_action_xyz(step, observation=current_obs, action_angle=current_action, desired=desired)

    obs, reward, done, _, info = env.step(action)
    current_obs = obs
    current_action = action

    if done:
        env.reset()

    step += 1