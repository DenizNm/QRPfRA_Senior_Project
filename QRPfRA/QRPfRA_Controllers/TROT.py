import numpy as np
from gymnasium.utils.env_checker import check_env
import QRPfRA
import tensorflow as tf
from scipy.spatial.transform import Rotation as R


class TROT():
    def __init__(self, tf_lite_model_path, env_num_states, env_num_actions, steps=75, reverse=False, fr_height_diff=-0.02, left_leg_ctrl=0.0, right_leg_ctrl=0.0):
        self.steps = steps
        self.reverse = reverse
        self.fr_height_diff = fr_height_diff
        self.left_leg_ctrl = left_leg_ctrl
        self.right_leg_ctrl = right_leg_ctrl
        self.leg_model = tf.lite.Interpreter(model_path=tf_lite_model_path)
        self.leg_model.allocate_tensors()
        self.prev_y_list = np.zeros(int(steps/10))
        self.current_action_angle = np.zeros(12).flatten()
        self.step = 0
        self.current_observation = np.zeros(env_num_states)
        self.num_actions = env_num_actions
        self.in_step_cnt = 0
        self.generated_action = self._trot_generation()


    def standUp(self):
        front_left_leg = np.array([-0.05, 0.0, -0.1])
        back_left_leg = np.array([-0.05, 0.0, -0.105])

        front_right_leg = np.array([-0.05, 0.0, -0.1])
        back_right_leg = np.array([-0.05, 0.0, -0.105])
        return np.concatenate((front_left_leg, back_left_leg, front_right_leg, back_right_leg))


    def _generate_points(self, input_list, total_step):
        step_between_points = total_step // (len(input_list) - 1)
        output = []
        for i in range(len(input_list) - 1):
            output.extend(np.linspace(input_list[i], input_list[i + 1], step_between_points))
        return output

    def _trot_generation(self):
        x_keyframes_d1 = [-0.05, -0.05, -0.05, -0.05]
        y_keyframes_d1 = [0.0, -0.05, 0.0, 0.05]
        z_keyframes_d1 = [-0.12, -0.15, -0.1, -0.15]

        x_keyframes_d2 = [-0.05, -0.05, -0.05, -0.05]
        y_keyframes_d2 = [0.0, 0.05, 0.0, -0.05]
        z_keyframes_d2 = [-0.1, -0.15, -0.12, -0.15]

        # Interpolate as much as the number of steps
        x_d1 = self._generate_points(x_keyframes_d1, self.steps + 1)
        y_d1 = self._generate_points(y_keyframes_d1, self.steps + 1)
        z_d1 = self._generate_points(z_keyframes_d1, self.steps + 1)

        x_d2 = self._generate_points(x_keyframes_d2, self.steps + 1)
        y_d2 = self._generate_points(y_keyframes_d2, self.steps + 1)
        z_d2 = self._generate_points(z_keyframes_d2, self.steps + 1)

        front_left_rear_right = np.array([x_d1, y_d1, z_d1])
        front_right_rear_left = np.array([x_d2, y_d2, z_d2])

        # Adaptive control of the legs
        left_leg_x = [0.0, 0.0, 0.0, 0.0]
        left_leg_y = [0.0, -self.left_leg_ctrl, 0.0, self.left_leg_ctrl]
        left_leg_z = [0.0, 0.0, 0.0, 0.0]

        right_leg_x = [0.0, 0.0, 0.0, 0.0]
        right_leg_y = [0.0, -self.right_leg_ctrl, 0.0, self.right_leg_ctrl]
        right_leg_z = [0.0, 0.0, 0.0, 0.0]

        left_leg_x = self._generate_points(left_leg_x, self.steps + 1)
        left_leg_y = self._generate_points(left_leg_y, self.steps + 1)
        left_leg_z = self._generate_points(left_leg_z, self.steps + 1)

        right_leg_x = self._generate_points(right_leg_x, self.steps + 1)
        right_leg_y = self._generate_points(right_leg_y, self.steps + 1)
        right_leg_z = self._generate_points(right_leg_z, self.steps + 1)

        if not self.reverse:
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
        trot_action[:, 5] = trot_action[:, 5] + self.fr_height_diff
        trot_action[:, 11] = trot_action[:, 11] + self.fr_height_diff

        # Clip the z values
        trot_action[:, 2] = np.clip(trot_action[:, 2], -0.195, -0.1)
        trot_action[:, 5] = np.clip(trot_action[:, 5], -0.195, -0.1)
        trot_action[:, 8] = np.clip(trot_action[:, 8], -0.195, -0.1)
        trot_action[:, 11] = np.clip(trot_action[:, 11], -0.195, -0.1)

        # Clip the x values
        trot_action[:, 0] = np.clip(trot_action[:, 0], -0.09, 0.01)
        trot_action[:, 3] = np.clip(trot_action[:, 3], -0.09, 0.01)
        trot_action[:, 6] = np.clip(trot_action[:, 6], -0.09, 0.01)
        trot_action[:, 9] = np.clip(trot_action[:, 9], -0.09, 0.01)

        return trot_action


    def _run_inference(self, model, input_data):
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)

        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        output_data = model.get_tensor(output_details[0]['index'])
        return output_data

    def _phase_transition_controller(self, act_in, act_out):
        err = act_out - act_in
        action = act_in + 0.08 * err
        return action

    def _action_to_xyz_angle(self, action_xyz, current_action_angle):
        if type(action_xyz) != list:
            action_xyz = action_xyz.tolist()
        elif type(action_xyz) == list:
            pass
        else:
            raise ValueError("Invalid data type for action_xyz")

        FL = self._run_inference(self.leg_model, action_xyz[0:3])
        RL = self._run_inference(self.leg_model, action_xyz[3:6])

        FR = self._run_inference(self.leg_model, action_xyz[6:9])
        RR = self._run_inference(self.leg_model, action_xyz[9:12])

        action_angle = np.array([FL, RL, FR, RR]).flatten()

        action_angle = self._phase_transition_controller(current_action_angle, action_angle)
        return action_angle

    def _update_y_list(self, y_list, y):
        y_list = np.roll(y_list, 1)
        y_list[0] = y
        return y_list, np.mean(y_list)

    def _get_orientation(self, observation):
        # TODO: Obtain the orientation from accelerometer and gyroscope, delete framequat in xml file
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

    def get_action_xyz(self, current_step, observation, action_angle, desired=0):
        self.current_action_angle = action_angle
        self.current_observation = observation


        self.step = current_step
        if current_step > 100:
            action_xyz = self.generated_action[self.in_step_cnt, :]
            if self.in_step_cnt == (self.steps - 2):
                self.in_step_cnt = 0

            self.in_step_cnt += 1

            control_y_direction = self._get_orientation(self.current_observation)
            prev_y_list, mean_y_direction = self._update_y_list(self.prev_y_list, control_y_direction)
            print(f"Control y direction: {mean_y_direction}")

            error = (mean_y_direction - desired) * 2

            # print(f"Control y direction: {mean_y_direction}, Error: {error}")
            if error > 0:
                self.left_leg_ctrl = 0.01 * abs(error)
                self.right_leg_ctrl = 0
                self.fr_height_diff = -0.02
                self.generated_action = self._trot_generation()
            else:
                self.left_leg_ctrl = 0.0
                self.right_leg_ctrl = 0.01 * abs(error)
                self.fr_height_diff = -0.02
                self.generated_action = self._trot_generation()

        elif current_step <= 100:
            mean_y_direction = 0
            action_xyz = self.standUp()

        action_angle = self._action_to_xyz_angle(action_xyz, self.current_action_angle)
        return action_angle

