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
model_load = True
#model_weight_path = "./save_model/.../dqn_weight_00.h5"
model_weight_path = "./save_model/best_weight.h5"


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
            dict(throttle=0.8, steering=0.1),
            dict(throttle=0.8, steering=-0.1),
            dict(throttle=0.6, steering=0.2),
            dict(throttle=0.6, steering=-0.2),
            dict(throttle=0.6, steering=0.3),
            dict(throttle=0.6, steering=-0.3),
            dict(throttle=1.0, steering=0)
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
            return -1.0

        return 0.0

    def calc_dist_reward_value(self, sensing_info):
        # reward_value = math.exp(-(max(abs(sensing_info.to_middle)-2.0, 0.0))*1.2)
        # reward_value = math.exp(-(max(abs(sensing_info.to_middle)-1.0, 0.0))*1.2)
        reward_value = math.exp(-(abs(sensing_info.to_middle))*1.2)

        MARGIN = 0.15
        OBSTACLE_WIDTH = 2.00
        CAR_WIDTH = 2.50

        CAR_OBSTACLE_MIN_DIST = (OBSTACLE_WIDTH + CAR_WIDTH)/2.00

        SPEED_DIST_RATE = 30.0/30.0 # 30km/h 일 때30m

        speed = sensing_info.speed
        tfo = sensing_info.track_forward_obstacles

        for o_dist, o_center_dist in tfo:
            min_dist = max(SPEED_DIST_RATE*speed, 1.0)
            danger_o_dist = CAR_OBSTACLE_MIN_DIST - (CAR_OBSTACLE_MIN_DIST*o_dist/min_dist)
            if abs(sensing_info.to_middle - o_center_dist) < danger_o_dist:
                reward_value -= 1.0
                break

        return reward_value

    def calc_speed_reward_value(self, sensing_info):
        goal_speed = 100
        speed = sensing_info.speed
        reward_value = speed/goal_speed

        return reward_value

    def calc_angle_reward_value(self, sensing_info):
        ma = sensing_info.moving_angle
        tfa = sensing_info.track_forward_angles
        tfa_differences = []

        # first_curve_dist = 0
        # first_curve_angle = 0
        thresh_angle = 20

        i = 1
        while i < len(tfa):
            # diff_angle = tfa[i] - tfa[i-1]
            # tfa_differences.append(abs(diff_angle))
            # if first_curve_angle == 0 and abs(diff_angle) > thresh_angle:
            #     first_curve_dist = i - 1
            #     first_curve_angle = diff_angle
            tfa_differences.append(abs(tfa[i] - tfa[i-1]))
            i = i + 1

        max_diff_angle = max(tfa_differences)
        max_angle_dist = tfa_differences.index(max_diff_angle)        
        max_angle = tfa[max_angle_dist]

        i = max_angle_dist
        if tfa[i + 1] - tfa[i] < 0:
            max_angle = max_angle * -1

        reward_value = 0

        if max_diff_angle < 5:
            reward_value = round(math.exp(-max(abs(ma) - 3, 0)), 2)
        else:
            if max_angle_dist == 0:
                reward_value = 1.0
            # elif 1 <= first_curve_dist < 3:
            #     reward_value = 1.0 - min(abs(first_curve_angle - ma), thresh_angle)/thresh_angle
            elif 1 <= max_angle_dist < 3:
                reward_value = 1.0 - min(abs(max_angle - ma), thresh_angle)/thresh_angle
                # reward_value = 1.0 - abs(max_angle - ma)/thresh_angle
            else:
                reward_value = 1.0 - min(abs(tfa[0] - ma), thresh_angle)/thresh_angle
                # reward_value = 1.0 - abs(tfa[0] - ma)/thresh_angle

        return abs(reward_value)

    def calc_thresh_angle_reward(self, sensing_info):
        ANGLE_REWARD = 0
        TRACK_OUTLINE = 10
        DISTANCE_DECAY_RATE = 0.8
        dist_to_outline = abs(sensing_info.to_middle) - TRACK_OUTLINE

        RIGHT_AVOID_ANGLE = sensing_info.to_middle > 5 and sensing_info.moving_angle > 0
        LEFT_AVOID_ANGLE = sensing_info.to_middle < -5 and sensing_info.moving_angle < 0
        if RIGHT_AVOID_ANGLE or LEFT_AVOID_ANGLE:
            ANGLE_REWARD = math.exp(-(abs(dist_to_outline) * DISTANCE_DECAY_RATE))

        return (round(ANGLE_REWARD,1) * -1)


    # =========================================================== #
    # Reward Function
    # =========================================================== #
    def compute_reward(self, sensing_info):

        # =========================================================== #
        # Area for writing code
        # =========================================================== #
        # Editing area starts from here
        #
        fc = self.failure_condition(sensing_info) * 4.0
        dist_reward_value = self.calc_dist_reward_value(sensing_info)
        thresh_reward_value = self.calc_thresh_angle_reward(sensing_info)
        angle_reward_value = 0
        speed_reward_value = 0

        if 0.5 < dist_reward_value:
            angle_reward_value = self.calc_angle_reward_value(sensing_info)

        if 1.0 < dist_reward_value + angle_reward_value:
            speed_reward_value = self.calc_speed_reward_value(sensing_info)

        reward = speed_reward_value + dist_reward_value + angle_reward_value + fc + thresh_reward_value

        # print(f"[Reward]{reward:0.3f} [to_middle]{round(self.sensing_info.to_middle,2)}, D:{dist_reward_value:0.3f} [angle]track:{self.sensing_info.track_forward_angles[0]} A:{angle_reward_value:0.3f} T:{thresh_reward_value:0.3f} [etc]S:{speed_reward_value:0.3f}")
        print(f"[Reward]{reward:0.3f} [to_middle]{round(self.sensing_info.to_middle,2)}, D:{dist_reward_value:0.3f} \
            [angle]track:{self.sensing_info.track_forward_angles[0]} angle:{self.sensing_info.moving_angle} A:{angle_reward_value:0.3f} T:{thresh_reward_value:0.3f} \
            [etc]speed:{self.sensing_info.speed} S:{speed_reward_value:0.3f}")        
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
