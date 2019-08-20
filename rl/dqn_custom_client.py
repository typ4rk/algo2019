import setup_path
from dqn_model import DQNClient
from dqn_model import DQNParam
import math
import numpy
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
            dict(throttle=0.7, steering=0.2),
            dict(throttle=0.7, steering=-0.2),
            dict(throttle=0.6, steering=0.3),
            dict(throttle=0.6, steering=-0.3),
            dict(throttle=1.0, steering=0)
        ]
        #
        # Editing area ends
        # ==========================================================#
        return actions

    def get_track_forward_angle_differences(self, sensing_info):
        tfa = sensing_info.track_forward_angles
        tfa_differences = []

        i = 1
        while i < len(tfa):
            tfa_differences.append(tfa[i - 1] - tfa[i])
            i = i + 1

        return tfa_differences

    def get_guide_line(self, sensing_info):
        tfa = sensing_info.track_forward_angles
        tfad = self.get_track_forward_angle_differences(sensing_info)
        tfad_std = numpy.std(tfad)

        PRED_STARTING_DIST = 5 # 50m
        PRED_FATHEST_DIST = len(tfa) - 1
        PRED_RANGE = PRED_FATHEST_DIST - PRED_STARTING_DIST
        WEIGHT_ON_FATHEST_DIST = 0.1
        WEIGHT_ON_STARTING_DIST = 1.0

        WEIGHT_DIFF = WEIGHT_ON_STARTING_DIST-WEIGHT_ON_FATHEST_DIST
        DEC_RATE = WEIGHT_DIFF/PRED_RANGE
        WEIGHT_ON_0 = WEIGHT_ON_FATHEST_DIST + (WEIGHT_DIFF * PRED_FATHEST_DIST / PRED_RANGE)

        i = PRED_FATHEST_DIST

        guide_line = 0.0
        while PRED_STARTING_DIST <= i:
            weight = -DEC_RATE*i + WEIGHT_ON_0
            cur_guide_line = -min(tfa[i]*weight, 90)*(self.half_road_limit-2.0)/90
            guide_line = cur_guide_line if cur_guide_line*guide_line < 0 or guide_line < cur_guide_line else guide_line
            i -= 1

        print(f"tfa[5]: {tfa[5]:0.3f} tfad_std:{tfad_std:0.3f} guide_line:{guide_line:0.3f}")

        return guide_line

    def failure_condition(self, sensing_info):
        thresh_dist = self.half_road_limit  # 4 wheels off the track
        dist = abs(sensing_info.to_middle)
        conds = (thresh_dist < dist, sensing_info.collided)

        if any(conds):
            return -1.0

        return 0.0

    def calc_dist_reward_value(self, sensing_info):
        baseline = self.get_guide_line(sensing_info)
        reward_value = math.exp(-max(abs(baseline - sensing_info.to_middle)-1.0, 0.0)*1.2)

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
        tfad = self.get_track_forward_angle_differences(sensing_info)

        thresh_angle = 20

        max_diff_angle = max(tfad)
        max_angle_dist = tfad.index(max_diff_angle)
        max_angle = tfa[max_angle_dist];

        if max_angle_dist == 0:
            reward_value = 1.0
        elif 1 <= max_angle_dist < 3:
            reward_value = 1.0 - (max_angle - ma)/thresh_angle
        else:
            reward_value = 1.0 - (tfa[0] - ma)/thresh_angle

        return abs(reward_value)

    def calc_thresh_angle_reward(self, sensing_info):
        ANGLE_REWARD = 0
        TRACK_OUTLINE = 10
        DISTANCE_DECAY_RATE = 0.8
        dist_to_outline = abs(sensing_info.to_middle) - TRACK_OUTLINE

        RIGHT_AVOID_ANGLE = sensing_info.to_middle > 5 and sensing_info.moving_angle > 0
        LEFT_AVOID_ANGLE = sensing_info.to_middle < -5 and sensing_info.moving_angle < 0
        if RIGHT_AVOID_ANGLE == True or LEFT_AVOID_ANGLE == True:
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

        print(f"lap_progress: {sensing_info.lap_progress} d:{dist_reward_value:0.3f} a:{angle_reward_value:0.3f} s:{speed_reward_value:0.3f} t:{thresh_reward_value:0.3f} f:{fc} = {reward:0.3f}")
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
