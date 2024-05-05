import numpy as np
from gymnasium.utils.env_checker import check_env
import QRPfRA
import tensorflow as tf
from scipy.spatial.transform import Rotation as R
import pygame

env = QRPfRA.QRPfRA_v3(raspberry_pi=True)
check_env(env)
obs = env.reset()
#env.render_mode = "human"


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1, 1))


num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

right_leg_model = tf.lite.Interpreter(model_path='/home/deniz/vscode_projects/QRPfRA_Senior_Project/QRPfRA/IK_Models/left_leg_model_quantized.tflite')
right_leg_model.allocate_tensors()

left_leg_model = tf.lite.Interpreter(model_path='/home/deniz/vscode_projects/QRPfRA_Senior_Project/QRPfRA/IK_Models/left_leg_model_quantized.tflite')
left_leg_model.allocate_tensors()

def init_position():
    front_left_leg = np.array([-0.05, 0.0, -0.1])
    back_left_leg = np.array([-0.05, 0.0, -0.105])

    front_right_leg = np.array([-0.05, 0.0, -0.1])
    back_right_leg = np.array([-0.05, 0.0, -0.105])
    return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))


def second_pose():
    front_left_leg = np.array([-0.15, 0, -0.12])
    back_left_leg = np.array([-0.15, 0, -0.125])

    front_right_leg = np.array([0.15, 0, -0.12])
    back_right_leg = np.array([0.15, 0, -0.125])
    return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))


def third_pose():
    front_left_leg = np.array([-0.05, -0.12, -0.15])
    back_left_leg = np.array([-0.05, -0.12, -0.155])

    front_right_leg = np.array([-0.05, -0.12, -0.15])
    back_right_leg = np.array([-0.05, -0.12, -0.155])
    return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))


def generate_points(input_list, total_step):
    step_between_points = total_step // (len(input_list) - 1)
    output = []
    for i in range(len(input_list) - 1):
        output.extend(np.linspace(input_list[i], input_list[i + 1], step_between_points))
    return output


def trot_generation(steps=100, reverse=False, fr_height_diff=0.0, left_leg_ctrl=0, right_leg_ctrl=0):
    x_keyframes_d1 = [-0.05, -0.05, -0.05]
    y_keyframes_d1 = [-0.04, 0.04, -0.04]
    z_keyframes_d1 = [-0.12, -0.1, -0.12]

    x_keyframes_d2 = [-0.05, -0.05, -0.05]
    y_keyframes_d2 = [0.04, -0.04, 0.04]
    z_keyframes_d2 = [-0.1, -0.12, -0.1]

    # Interpolate as much as the number of steps
    x_d1 = generate_points(x_keyframes_d1, steps+1)
    y_d1 = generate_points(y_keyframes_d1, steps+1)
    z_d1 = generate_points(z_keyframes_d1, steps+1)

    x_d2 = generate_points(x_keyframes_d2, steps+1)
    y_d2 = generate_points(y_keyframes_d2, steps+1)
    z_d2 = generate_points(z_keyframes_d2, steps+1)

    front_left_rear_right = np.array([x_d1, y_d1, z_d1])
    front_right_rear_left = np.array([x_d2, y_d2, z_d2])

    # Adaptive control of the legs
    left_leg_x = [0.0, 0.0, 0.0]
    left_leg_y = [0.0, -left_leg_ctrl, 0.0]
    left_leg_z = [0.0, 0.0, 0.0]

    right_leg_x = [0.0, 0.0, 0.0]
    right_leg_y = [0.0, -right_leg_ctrl, 0.0]
    right_leg_z = [0.0, 0.0, 0.0]

    left_leg_x = generate_points(left_leg_x, steps+1)
    left_leg_y = generate_points(left_leg_y, steps+1)
    left_leg_z = generate_points(left_leg_z, steps+1)

    right_leg_x = generate_points(right_leg_x, steps+1)
    right_leg_y = generate_points(right_leg_y, steps+1)
    right_leg_z = generate_points(right_leg_z, steps+1)

    if not reverse:
        trot_action = np.concatenate(
            (front_left_rear_right + left_leg_y, front_right_rear_left + left_leg_y,
             front_right_rear_left + right_leg_y, front_left_rear_right + right_leg_y)).T
    elif reverse:
        trot_action = np.concatenate(
            (front_left_rear_right + left_leg_y, front_right_rear_left + left_leg_y,
             front_right_rear_left + right_leg_y, front_left_rear_right + right_leg_y)).T

    # Clip the y values
    trot_action[:, 1] = np.clip(np.add(trot_action[:, 1], left_leg_y), -0.1, 0.1)
    trot_action[:, 4] = np.clip(np.add(trot_action[:, 4], left_leg_y), -0.1, 0.1)
    trot_action[:, 7] = np.clip(np.add(trot_action[:, 7], right_leg_y), -0.1, 0.1)
    trot_action[:, 10] = np.clip(np.add(trot_action[:, 10], right_leg_y), -0.1, 0.1)

    # Add the height difference
    trot_action[:, 5] = trot_action[:, 5] + fr_height_diff
    trot_action[:, 11] = trot_action[:, 11] + fr_height_diff

    # Clip the z values
    trot_action[:, 2] = np.clip(trot_action[:, 2], -0.195, -0.1)  # L
    trot_action[:, 5] = np.clip(trot_action[:, 5], -0.195, -0.1)  # L
    trot_action[:, 8] = np.clip(trot_action[:, 8], -0.195, -0.1)  # R
    trot_action[:, 11] = np.clip(trot_action[:, 11], -0.195, -0.1)  # R

    # Clip the x values
    trot_action[:, 0] = np.clip(trot_action[:, 0], -0.09, 0.01)  # L
    trot_action[:, 3] = np.clip(trot_action[:, 3], -0.09, 0.01)  # L
    trot_action[:, 6] = np.clip(trot_action[:, 6], -0.09, 0.01)  # R
    trot_action[:, 9] = np.clip(trot_action[:, 9], -0.09, 0.01)  # R

    return trot_action



def trot_V2(steps=100, reverse=False, fr_height_diff=0.0, left_leg_ctrl=0, right_leg_ctrl=0):
    FL_x = np.ones(4) * -0.05
    FL_y = np.array([(left_leg_ctrl + 0.04), -(left_leg_ctrl + 0.02),  -(left_leg_ctrl + 0.02),  (left_leg_ctrl + 0.04)])
    FL_z = np.array([-(0.14 - left_leg_ctrl/10), -(0.16 + left_leg_ctrl/5), -(0.16 - left_leg_ctrl/5), -(0.13 + left_leg_ctrl/10)])

    RL_x = np.ones(4) * -0.05
    RL_y = np.array([-(left_leg_ctrl + 0.02),  (left_leg_ctrl + 0.04), (left_leg_ctrl + 0.04), -(left_leg_ctrl + 0.02)])
    RL_z = np.array([-(0.18 + left_leg_ctrl/5), -(0.13 - left_leg_ctrl/10), -(0.18 + left_leg_ctrl/5), -(0.14 - left_leg_ctrl/10)]) - 0.02

    FR_x = np.ones(4) * -0.05
    FR_y = np.array([-(right_leg_ctrl+0.02),  (right_leg_ctrl + 0.04), (right_leg_ctrl + 0.04), -(right_leg_ctrl + 0.02)])
    FR_z = np.array([-(0.18 + right_leg_ctrl/5), -(0.13 - right_leg_ctrl/10), -(0.18 + right_leg_ctrl/5), -(0.14 - right_leg_ctrl/10)])

    RR_x = np.ones(4) * -0.05
    RR_y = np.array([(right_leg_ctrl + 0.04),  -(right_leg_ctrl + 0.02), -(right_leg_ctrl + 0.02),  (right_leg_ctrl + 0.04)])
    RR_z = np.array([-(0.14 - right_leg_ctrl/10), -(0.16 + right_leg_ctrl/5), -(0.16 - right_leg_ctrl/5), -(0.13 + right_leg_ctrl/10)]) - 0.02


    if not reverse:
        FL = [FL_x, FL_y, FL_z]
        RL = [RL_x, RL_y, RL_z]
        FR = [FR_x, FR_y, FR_z]
        RR = [RR_x, RR_y, RR_z]
    elif reverse:
        FL = [FL_x, RL_y, FL_z]
        RL = [RL_x, FL_y, RL_z]
        FR = [FR_x, RR_y, FR_z]
        RR = [RR_x, FR_y, RR_z]

    FL_generated = [generate_points(coordination, steps+1) for coordination in FL]
    RL_generated = [generate_points(coordination, steps+1) for coordination in RL]
    FR_generated = [generate_points(coordination, steps+1) for coordination in FR]
    RR_generated = [generate_points(coordination, steps+1) for coordination in RR]

    FL_generated[:][0] = np.clip(FL_generated[:][0], -0.09, 0.01)
    RL_generated[:][0] = np.clip(RL_generated[:][0], -0.09, 0.01)
    FR_generated[:][0] = np.clip(FR_generated[:][0], -0.09, 0.01)
    RR_generated[:][0] = np.clip(RR_generated[:][0], -0.09, 0.01)

    FL_generated[:][1] = np.clip(FL_generated[:][1], -0.06, 0.06)
    RL_generated[:][1] = np.clip(RL_generated[:][1], -0.06, 0.06)
    FR_generated[:][1] = np.clip(FR_generated[:][1], -0.06, 0.06)
    RR_generated[:][1] = np.clip(RR_generated[:][1], -0.06, 0.06)

    FL_generated[:][2] = np.clip(FL_generated[:][2], -0.195, -0.09)
    RL_generated[:][2] = np.clip(RL_generated[:][2], -0.195, -0.09)
    FR_generated[:][2] = np.clip(FR_generated[:][2], -0.195, -0.09)
    RR_generated[:][2] = np.clip(RR_generated[:][2], -0.195, -0.09)

    foot_list = np.concatenate((FL_generated, RL_generated, FR_generated, RR_generated)).T

    return foot_list






def _run_inference(model, input_data):
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)

    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data

def phase_transition_controller(act_in, act_out):
    err = act_out - act_in
    action = act_in + 0.10 * err
    return action


def action_xyz_to_angle(action_xyz, current_action_angle):
    if type(action_xyz) != list:
        action_xyz = action_xyz.tolist()
    elif type(action_xyz) == list:
        pass
    else:
        raise ValueError("Invalid data type for action_xyz")

    FL = _run_inference(left_leg_model, action_xyz[0:3])
    RL = _run_inference(left_leg_model, action_xyz[3:6])

    FR = _run_inference(right_leg_model, action_xyz[6:9])
    RR = _run_inference(right_leg_model, action_xyz[9:12])

    action_angle = np.array([FL, RL, FR, RR]).flatten()

    action_angle = phase_transition_controller(current_action_angle, action_angle)
    return action_angle


def get_orientation(observation):
    #TODO: Obtain the orientation from accelerometer and gyroscope, delete framequat in xml file
    rotation = R.from_quat(observation[21:])
    # print(rotation)
    coordinates = "YZX"
    euler_angles = rotation.as_euler(coordinates, degrees=True)
    if euler_angles[2] <= 0:
        euler_angles[2] = -(euler_angles[2] + 180)
    else:
        euler_angles[2] = 180 - euler_angles[2]

    control_y_direction = euler_angles[2]
    return control_y_direction

def update_y_list(y_list, y):
    y_list = np.roll(y_list, 1)
    y_list[0] = y
    return y_list, np.mean(y_list)


current_action_angle = np.zeros(12).flatten()
step = 0
left_leg_ctrl = 0.0
right_leg_ctrl = 0.0

total_steps = 50
generated_action = trot_V2(total_steps, reverse=False, fr_height_diff=0.02, left_leg_ctrl=0.0, right_leg_ctrl=0.0)

prev_y_list = np.zeros(5)
current_observations = np.zeros(num_states)
in_step_cnt = 0

desired = 0
init_pos = False
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    keys = pygame.key.get_pressed()

    if step > 100 and not init_pos:
        action_xyz = generated_action[in_step_cnt, :]
        if in_step_cnt == (total_steps-2):
            in_step_cnt = 0

        in_step_cnt += 1

        control_y_direction = get_orientation(current_observations)
        prev_y_list, mean_y_direction = update_y_list(prev_y_list, control_y_direction)

        # Adjust desired angle using keyboard input from A bd G keys
        if keys[pygame.K_a]:
            desired += 1
        if keys[pygame.K_g]:
            desired -= 1


        desired = np.clip(desired, -178, 178)
        print(f"Desired: {desired}, Control y direction: {mean_y_direction}")

        error = (mean_y_direction - desired) * 0.0016
        if error > 0:
            generated_action = trot_V2(total_steps, reverse=False, fr_height_diff=-0.005,
                                               left_leg_ctrl= abs(error), right_leg_ctrl=-0.8 * abs(error))
        else:
            generated_action = trot_V2(total_steps, reverse=False, fr_height_diff=-0.01,
                                               left_leg_ctrl=-0.8 * abs(error), right_leg_ctrl= abs(error))

    elif step <= 100 or init_pos:
        action_xyz = init_position()


    if keys[pygame.K_p]:
        if init_pos:
            init_pos = False
        else:
            init_pos = True

    
    action_angle = action_xyz_to_angle(action_xyz, current_action_angle)
    obs, reward, done, _, info = env.step(action_angle)
    current_action_angle = action_angle
    current_observations = obs

    if done:
        env.reset()

    step += 1
