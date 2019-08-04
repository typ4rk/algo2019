import setup_path
from dqn_model import DQNClient
from dqn_model import DQNParam
import math
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import sys

# =========================================================== #
# Training finish conditions (hour)
# assign training duration by hour : 0(limit less), 1 (an hour), 1.5 (an hour and half) ...
# =========================================================== #
training_duration = 0

# =========================================================== #
# model/weight load option
# =========================================================== #
model_load = False
model_weight_path = "./save_model/.../dqn_weight_00.h5"


# ===========================================================

class DQNCustomClient(DQNClient):
    def __init__(self):
        self.dqn_param = self.make_dqn_param()
        super().__init__(self.dqn_param)

    # =========================================================== #
    # Tuning area (Hyper-parameters for model training)
    # =========================================================== #
    @staticmethod
    def make_dqn_param():
        dqn_param = DQNParam()
        dqn_param.discount_factor = 0.99
        dqn_param.learning_rate = 0.00025
        dqn_param.epsilon = 1.0
        dqn_param.epsilon_decay = 0.999
        dqn_param.epsilon_min = 0.01
        dqn_param.batch_size = 100
        dqn_param.train_start = 1000
        dqn_param.memory_size = 20000
        return dqn_param

    # =========================================================== #
    # Action Space (Control Panel)
    # =========================================================== #
    def action_space(self):
        # =========================================================== #
        # Area for writing code
        # =========================================================== #
        # Editing area starts from here
        #
        actions = [
            dict(throttle=0.6, steering=0.1),
            dict(throttle=0.6, steering=-0.1),
            dict(throttle=0.6, steering=0.2),
            dict(throttle=0.6, steering=-0.2),
            dict(throttle=0.6, steering=0)
        ]
        #
        # Editing area ends
        # ==========================================================#
        return actions

    def failure_condition(self, sensing_info):
        thresh_dist = self.half_road_limit  # 4 wheels off the track
        dist = abs(sensing_info.to_middle)
        conds = (thresh_dist < dist
                , sensing_info.collided);

        if any(conds):
            return True

        return False

    def calc_dist_reward_value(self, sensing_info):
        thresh_dist = self.half_road_limit  # 4 wheels off the track
        dist = abs(sensing_info.to_middle)
        reward_value = (thresh_dist-dist)/thresh_dist

        tfo = sensing_info.track_forward_obstacles

        if tfo:
            max_dist_to_o = 3.0
            o_dist, o_center_dist = tfo[0]
            if o_dist < 30:
                o_reward_value = 0.0
                if o_center_dist < 0:
                    best = o_center_dist + 1.0 + 1.40
                    abs_diff = abs((dist-best) if (dist-best) < max_dist_to_o else max_dist_to_o)
                    if best <= dist:
                        o_reward_value = 1.0 - abs_diff/max_dist_to_o
                    else:
                        o_reward_value = -0.5 * abs_diff/max_dist_to_o

                else:
                    best = o_center_dist - 1.0 - 1.40
                    abs_diff = abs((dist-best) if (dist-best) < max_dist_to_o else max_dist_to_o)
                    if dist <= best:
                        o_reward_value = 1.0 - abs_diff/max_dist_to_o
                    else:
                        o_reward_value = -0.5 * abs_diff/max_dist_to_o
                reward_value = o_reward_value

        return reward_value

    def calc_speed_reward_value(self, sensing_info):
        max_speed = 80
        speed = sensing_info.speed
        reward_value = speed/max_speed if speed/max_speed <= 1.0 else 1.0

        return reward_value

    def calc_angle_reward_value(self, sensing_info):
        ma = sensing_info.moving_angle
        tfa = sensing_info.track_forward_angles
        tfa_differences = []

        i = 1;
        while i < len(tfa):
            tfa_differences.append(tfa[i - 1] - tfa[i])
            i = i + 1

        thresh_angle = 20

        max_diff_angle = max(tfa_differences)
        max_angle_dist = tfa_differences.index(max_diff_angle)
        max_angle = tfa[max_angle_dist];

        if max_angle_dist == 0:
            reward_value = 1.0
        elif 1 <= max_angle_dist < 3:
            reward_value = 1.0 - (max_angle - ma)/thresh_angle
        else:
            reward_value = 1.0 - (tfa[0] - ma)/thresh_angle

        return abs(reward_value)

    # =========================================================== #
    # Reward Function
    # =========================================================== #
    def compute_reward(self, sensing_info):

        # =========================================================== #
        # Area for writing code
        # =========================================================== #
        # Editing area starts from here
        #
        fc = self.failure_condition(sensing_info)
        speed_reward_value = self.calc_speed_reward_value(sensing_info)
        dist_reward_value = self.calc_dist_reward_value(sensing_info)
        angle_reward_value = self.calc_angle_reward_value(sensing_info)

        reward = speed_reward_value * dist_reward_value * angle_reward_value - (1.0 if fc else 0.0)

        print(f"sensing_info.lap_progress: {sensing_info.lap_progress} s:{speed_reward_value:0.3f} d:{dist_reward_value:0.3f} a:{angle_reward_value:0.3f} = {reward:0.3f}")
        #
        # Editing area ends
        # ==========================================================#
        return reward

    # =========================================================== #
    # Model network
    # =========================================================== #
    def build_custom_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()

        model.compile(loss='mse', optimizer=Adam(lr=self.dqn_param.learning_rate))

        return model


if __name__ == "__main__":
    client = DQNCustomClient()

    client.override_model()

    if model_load:
        client.agent.load_model(model_weight_path)

    client.run(training_duration)
    sys.exit()
