import numpy as np
from gymnasium.utils.env_checker import check_env
import QRPfRA
import tensorflow as tf

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

right_leg_model = tf.lite.Interpreter(model_path='/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/left_leg_model_quantized.tflite')
right_leg_model.allocate_tensors()

left_leg_model = tf.lite.Interpreter(model_path='/Users/deniz/PycharmProjects/QRPfRA_Senior_Project/QRPfRA/IK_Models/left_leg_model_quantized.tflite')
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


def trot_generation(steps=100, reverse=False, fr_height_diff=0.0):
    x_keyframes_d1 = [-0.05, -0.05, -0.05, -0.05]
    y_keyframes_d1 = [0.0, -0.08, 0.0, 0.08]
    z_keyframes_d1 = [-0.12, -0.13, -0.1, -0.13]

    x_keyframes_d2 = [-0.05, -0.05, -0.05, -0.05]
    y_keyframes_d2 = [0.0, 0.08, 0.0, -0.08]
    z_keyframes_d2 = [-0.1, -0.13, -0.13, -0.12]


    # Interpolate as much as the number of steps
    x_d1 = generate_points(x_keyframes_d1, steps+1)
    y_d1 = generate_points(y_keyframes_d1, steps+1)
    z_d1 = generate_points(z_keyframes_d1, steps+1)

    x_d2 = generate_points(x_keyframes_d2, steps+1)
    y_d2 = generate_points(y_keyframes_d2, steps+1)
    z_d2 = generate_points(z_keyframes_d2, steps+1)

    front_left_rear_right = np.array([x_d1, y_d1, z_d1])
    front_right_rear_left = np.array([x_d2, y_d2, z_d2])

    if not reverse:
        trot_action = np.concatenate(
            (front_left_rear_right, front_right_rear_left, front_right_rear_left, front_left_rear_right)).T
    elif reverse:
        trot_action = np.concatenate(
            (front_left_rear_right, front_right_rear_left, front_right_rear_left, front_left_rear_right)).T

    trot_action[:, 5] = trot_action[:, 5] + fr_height_diff
    trot_action[:, 11] = trot_action[:, 11] + fr_height_diff
    return trot_action


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
    action = act_in + 0.08 * err
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


current_action_angle = np.zeros(12).flatten()
step = 0


total_steps = 75
generated_action = trot_generation(total_steps, reverse=False, fr_height_diff=-0.02)
print("Shape is:", generated_action.shape)

in_step_cnt = 0
while True:
    if step > 100:
        action_xyz = generated_action[in_step_cnt, :]
        if in_step_cnt == (total_steps-2):
            in_step_cnt = 0

        in_step_cnt += 1

    elif step <= 100:
        action_xyz = init_position()


    action_angle = action_xyz_to_angle(action_xyz, current_action_angle)
    obs, reward, done, _, info = env.step(action_angle)
    current_action_angle = action_angle


    if done:
        env.reset()

    step += 1
