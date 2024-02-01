import numpy as np
from gymnasium.utils.env_checker import check_env
import QRPfRA

env = QRPfRA.QRPfRA_v3()
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
    front_left_leg = np.array([-0.045, 0, -0.10])
    back_left_leg = np.array([-0.045, 0, -0.10])

    front_right_leg = np.array([-0.045, 0, -0.10])
    back_right_leg = np.array([-0.045, 0, -0.10])
    return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))

def init_position2():
    front_left_leg = np.array([-0.045, 0, -0.19])
    back_left_leg = np.array([-0.045, 0, -0.19])

    front_right_leg = np.array([-0.045, 0, -0.19])
    back_right_leg = np.array([-0.045, 0, -0.19])
    return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))


def draw_elipse(a, b, count):
    x = np.linspace(-a, a, count)
    y = b * np.sqrt(1 - (x ** 2 / a ** 2))
    return x, y


a = 0.06
b = 0.06
count = 100
z, y = draw_elipse(a, b, count)


step = 0
while True:
    if step % 200 < 100:
        action = init_position()
    else:
        action = init_position2()

    """if step < 100:
        action = init_position()
    else:
        action = init_position2()
        index = step % count
        if index == 0:
            index = count - step % count - 1

        # FL
        action[1] = action[1] + y[index]
        action[2] = action[2] - z[index]

        # RL
        action[4] = action[4] - y[index]
        action[5] = action[5] + z[index]

        # FR
        action[7] = action[7] - y[index]
        action[8] = action[8] + z[index]

        # RR
        action[10] = action[10] + y[index]
        action[11] = action[11] - z[index]"""

    obs, reward, done, _, info = env.step(action)

    if done:
        env.reset()

    step += 1


"""
        if foot_observation[0] != 0 or foot_observation[3] != 0:
            action[1] = action[1] + y[step % count]
            action[2] = action[2]

            action[10] = action[10] + y[step % count]
            action[11] = action[11]

            action[4] = action[4] - y[step % count]
            action[5] = action[5] - z[step % count]

            action[7] = action[7] - y[step % count]
            action[8] = action[8] - z[step % count]

        elif foot_observation[1] != 0 or foot_observation[2] != 0:
            action[4] = action[4] - y[step % count]
            action[5] = action[5]

            action[7] = action[7] - y[step % count]
            action[8] = action[8]

            action[1] = action[1] + y[step % count]
            action[2] = action[2] + z[step % count]

            action[10] = action[10] + y[step % count]
            action[11] = action[11] + z[step % count]
        else:
"""
