import numpy as np
from gymnasium.utils.env_checker import check_env
import QRPfRA
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

class TROT:
    def __init__(self, tf_lite_model_path, env_num_actions, env_num_states):
        self.env = QRPfRA.QRPfRA_v3(use_serial_port=False)
        check_env(self.env)
        self.obs = self.env.reset()
        self.env.render_mode = "human"

        self.num_states = self.env.observation_space.shape[0]
        print("Size of State Space ->  {}".format(self.num_states))
        self.num_actions = self.env.action_space.shape[0]
        print("Size of Action Space ->  {}".format(self.num_actions))

        self.upper_bound = self.env.action_space.high[0]
        self.lower_bound = self.env.action_space.low[0]

        print("Max Value of Action ->  {}".format(self.upper_bound))
        print("Min Value of Action ->  {}".format(self.lower_bound))

        self.right_leg_model = tf.lite.Interpreter(model_path=tf_lite_model_path)
        self.right_leg_model.allocate_tensors()

        self.left_leg_model = tf.lite.Interpreter(model_path=tf_lite_model_path)
        self.left_leg_model.allocate_tensors()

        self.IMU_to_orient_model = tf.lite.Interpreter(model_path=tf_lite_model_path)
        self.IMU_to_orient_model.allocate_tensors()

        self.current_action_angle = np.zeros(12).flatten()
        self.step = 0

        self.left_leg_ctrl = 0.0
        self.right_leg_ctrl = 0.0

        self.total_steps = 75
        self.generated_action = self.trot_generation(self.total_steps, reverse=False, fr_height_diff=-0.02, left_leg_ctrl=0.0, right_leg_ctrl=0.0)

        self.prev_y_list = np.zeros(100)

        self.current_observations = np.zeros(self.num_states)
        self.in_step_cnt = 0


    def init_position(self):
        front_left_leg = np.array([-0.05, 0.0, -0.1])
        back_left_leg = np.array([-0.05, 0.0, -0.105])

        front_right_leg = np.array([-0.05, 0.0, -0.1])
        back_right_leg = np.array([-0.05, 0.0, -0.105])
        return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))

    def second_pose(self):
        front_left_leg = np.array([-0.15, 0, -0.12])
        back_left_leg = np.array([-0.15, 0, -0.125])

        front_right_leg = np.array([0.15, 0, -0.12])
        back_right_leg = np.array([0.15, 0, -0.125])
        return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))

    def third_pose(self):
        front_left_leg = np.array([-0.05, -0.12, -0.15])
        back_left_leg = np.array([-0.05, -0.12, -0.155])

        front_right_leg = np.array([-0.05, -0.12, -0.15])
        back_right_leg = np.array([-0.05, -0.12, -0.155])
        return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))

    def generate_points(self, input_list, total_step):
        step_between_points = total_step // (len(input_list) - 1)
        output = []
        for i in range(len(input_list) - 1):
            output.extend(np.linspace(input_list[i], input_list[i + 1], step_between_points))
        return output

    def trot_generation(self, steps=100, reverse=False, fr_height_diff=0.0, left_leg_ctrl=0, right_leg_ctrl=0):
        x_keyframes_d1 = [-0.05, -0.05, -0.05, -0.05]
        y_keyframes_d1 = [0.0, -0.05, 0.0, 0.05]
        z_keyframes_d1 = [-0.12, -0.15, -0.1, -0.15]

        x_keyframes_d2 = [-0.05, -0.05, -0.05, -0.05]
        y_keyframes_d2 = [0.0, 0.05, 0.0, -0.05]
        z_keyframes_d2 = [-0.1, -0.15, -0.12, -0.15]

        # Interpolate as much as the number of steps
        x_d1 = self.generate_points(x_keyframes_d1, steps + 1)
        y_d1 = self.generate_points(y_keyframes_d1, steps + 1)
        z_d1 = self.generate_points(z_keyframes_d1, steps + 1)

        x_d2 = self.generate_points(x_keyframes_d2, steps + 1)
        y_d2 = self.generate_points(y_keyframes_d2, steps + 1)
        z_d2 = self.generate_points(z_keyframes_d2, steps + 1)

        front_left_rear_right = np.array([x_d1, y_d1, z_d1])
        front_right_rear_left = np.array([x_d2, y_d2, z_d2])

        # Adaptive control of the legs
        left_leg_x = [0.0, 0.0, 0.0, 0.0]
        left_leg_y = [0.0, -left_leg_ctrl, 0.0, left_leg_ctrl]
        left_leg_z = [0.0, 0.0, 0.0, 0.0]

        right_leg_x = [0.0, 0.0, 0.0, 0.0]
        right_leg_y = [0.0, -right_leg_ctrl, 0.0, right_leg_ctrl]
        right_leg_z = [0.0, 0.0, 0.0, 0.0]

        left_leg_x = self.generate_points(left_leg_x, steps + 1)
        left_leg_y = self.generate_points(left_leg_y, steps + 1)
        left_leg_z = self.generate_points(left_leg_z, steps + 1)

        right_leg_x = self.generate_points(right_leg_x, steps + 1)
        right_leg_y = self.generate_points(right_leg_y, steps + 1)
        right_leg_z = self.generate_points(right_leg_z, steps + 1)

        if not reverse:
            trot_action = np.concatenate(
                (front_left_rear_right + left_leg_y, front_right_rear_left + left_leg_y,
                 front_right_rear_left + right_leg_y, front_left_rear_right + right_leg_y)).T
        else:
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

    def _run_inference(self, model, input_data):
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)

        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        output_data = model.get_tensor(output_details[0]['index'])
        return output_data

    def phase_transition_controller(self, act_in, act_out):
        err = act_out - act_in
        action = act_in + 0.08 * err
        return action

    def action_xyz_to_angle(self, action_xyz, current_action_angle):
        if type(action_xyz) != list:
            action_xyz = action_xyz.tolist()
        elif type(action_xyz) == list:
            pass
        else:
            raise ValueError("Invalid data type for action_xyz")

        FL = self._run_inference(self.left_leg_model, action_xyz[0:3])
        RL = self._run_inference(self.left_leg_model, action_xyz[3:6])

        FR = self._run_inference(self.right_leg_model, action_xyz[6:9])
        RR = self._run_inference(self.right_leg_model, action_xyz[9:12])

        action_angle = np.array([FL, RL, FR, RR]).flatten()

        action_angle = self.phase_transition_controller(current_action_angle, action_angle)
        return action_angle

    def get_orientation(self, observation, model):
        rotation = R.from_quat(observation[27:])
        # print(rotation)
        coordinates = "YZX"
        euler_angles = rotation.as_euler(coordinates, degrees=True)
        if euler_angles[2] <= 0:
            euler_angles[2] = -(euler_angles[2] + 180)
        else:
            euler_angles[2] = 180 - euler_angles[2]

        control_y_direction = euler_angles[2]
        return control_y_direction

    def update_y_list(self, y_list, y):
        y_list = np.roll(y_list, 1)
        y_list[0] = y
        return y_list, np.mean(y_list)

    def get_action_xyz(self, step, observation, action_angle, desired):
        if step > 100:
            action_xyz = self.generated_action[self.in_step_cnt, :]
            if self.in_step_cnt == (self.total_steps-2):
                self.in_step_cnt = 0

            self.in_step_cnt += 1

            control_y_direction = self.get_orientation(observation, self.IMU_to_orient_model)
            self.prev_y_list, mean_y_direction = self.update_y_list(self.prev_y_list, control_y_direction)

            error = (mean_y_direction - desired) * 0.1

            if error > 0:
                self.generated_action = self.trot_generation(self.total_steps, reverse=False, fr_height_diff=-0.02,
                                                   left_leg_ctrl=0.01 * abs(error), right_leg_ctrl=0)
            else:
                self.generated_action = self.trot_generation(self.total_steps, reverse=False, fr_height_diff=-0.02,
                                                   left_leg_ctrl=0, right_leg_ctrl=0.01 * abs(error))

        elif step <= 100:
            action_xyz = self.init_position()

        action_angle = self.action_xyz_to_angle(action_xyz, action_angle)
        return action_angle