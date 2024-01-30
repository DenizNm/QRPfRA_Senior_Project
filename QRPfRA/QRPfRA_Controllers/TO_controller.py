import numpy as np
from gymnasium.utils.env_checker import check_env
import QRPfRA

env = QRPfRA.QRPfRA_v3()
check_env(env)
obs = env.reset()
env.render_mode = "human"


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
    front_left_leg = np.array([-0.04, 0, -0.12])
    front_right_leg = np.array([0.04, 0, -0.12])

    back_left_leg = np.array([-0.04, 0, -0.12])
    back_right_leg = np.array([0.04, 0, -0.12])
    return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))

def init_position2():
    front_left_leg = np.array([-0.04, 0, -0.15])
    front_right_leg = np.array([0.04, 0, -0.15])

    back_left_leg = np.array([-0.04, 0, -0.15])
    back_right_leg = np.array([0.04, 0, -0.15])
    return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))



step = 0
while True:
    if step % 200 < 100:
        action = init_position()
    else:
        action = init_position2()


    obs, reward, done, info = env.step(action)
    #print("Euler Angles", quat_2_euler(obs[4:8]))


    if done:
        print("Episode finished after {} timesteps".format(i + 1))
        env.reset()

    step += 1

