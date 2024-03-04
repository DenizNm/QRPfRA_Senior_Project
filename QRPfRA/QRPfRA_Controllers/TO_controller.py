import numpy as np
from gymnasium.utils.env_checker import check_env
import QRPfRA

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


def quat_2_euler(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    #Convert to degrees
    roll = np.rad2deg(roll)
    pitch = np.rad2deg(pitch)
    yaw = np.rad2deg(yaw)

    return np.array([roll, pitch, yaw])


def init_position():
    front_left_leg = np.array([-0.025, -0.09, -0.15])
    back_left_leg = np.array([-0.025, -0.09, -0.155])

    front_right_leg = np.array([-0.025, -0.09, -0.15])
    back_right_leg = np.array([-0.025, -0.09, -0.155])
    return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))


step = 0
while True:
    action = init_position()

    obs, reward, done, _, info = env.step(action)

    if done:
        env.reset()

    step += 1

