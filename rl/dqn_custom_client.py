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
# Map: Mariana 
# episode: 1124  score: 11.2  check point reached: 11  lap: 1.14 [score]  47.9 / 4.19 % (= 11.4 ), episode: 388

# Map: Speed racing
# Try 1
# episode: 2959  score: 24.0  check point reached: 15  lap: 4.3 [score]  125.0 / 39.25 % (= 3.2 ), episode: 2487

# Try 2
# model_weight_path = "./save_model/dqn_weight_T0815_223314_2490_p39_speedracing_change_reward.h5"
# episode: 715  score: 13.1  check point reached: 18  lap: 5.11 [score]  90.5 / 24.73 % (= 3.7 ), episode: 362

# Try 3
# model_weight_path = "./save_model/dqn_weight_T0816_081323_370_p24_speedracing_change_reward.h5"
# episode: 1272  score: 40.9  check point reached: 46  lap: 12.63 [score]  98.4 / 31.99 % (= 3.1 ), episode: 797

# Try 4
# model_weight_path = "./save_model/dqn_weight_T0816_155459_800_p31_speedracing_change_reward.h5"
# episode: 1268  score: 49.6  check point reached: 40  lap: 11.02 [score]  367.3 / 100.0 % (= 3.7 ), episode: 1080

# Map: Mariana
# model_weight_path = "./save_model/dqn_weight_T0816_211827_1080_pass_finishline_speedracing.h5"

# T0818_133618 (action_space: -2~2)
# episode: 517  score: 33.28  check point reached: 32  lap: 4.19 [score]  51.65 / 5.96 % (= 8.7 ), episode: 462  vs.  [lap] 51.59 / 6.09 % (= 8.5 ), episode: 341

# Try 1 (action_space: -3~3)
# episode: 1540  score: 58.6  check point reached: 19  lap: 2.54 [score]  79.04 / 3.05 % (= 25.9 ), episode: 1114  vs.  [lap] 36.4 / 4.31 % (= 8.4 ), episode: 318
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
        dqn_param.learning_rate = 0.005 # 0.00025
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
            dict(throttle=0.5, steering=0.3),
            dict(throttle=0.5, steering=-0.3),
            dict(throttle=0.9, steering=0)
        ]
        #
        # Editing area ends
        # ==========================================================#
        return actions

    def compute_speed_reward(self, sensing_info):
        # ==========================================================#
        # Calculate speed reward for detecting curves 
        # ==========================================================#

        dist = abs(sensing_info.to_middle)
        up_speed = True
        reward = 0

        change_rate_angles = []
        for x in range(0, 9):
            change_rate = abs(sensing_info.track_forward_angles[x+1] - sensing_info.track_forward_angles[x])
            change_rate_angles.append(change_rate)
            if x < 5 and change_rate > 20:
                up_speed = False

        max_change_value = max(change_rate_angles)
        max_change_index = change_rate_angles.index(max_change_value)

        # in the max curve area
        if max_change_value > 15:
            if max_change_index < 4 and max_change_index > 0:
                up_speed = False
 
        if up_speed == False:
            down_speed_reward = 0.1
            if sensing_info.speed < 30:
                down_speed_reward = 0.2

            if dist > 2:
                reward = 0
            elif dist > 1:
                reward = down_speed_reward
            else:
                reward = down_speed_reward * 2

            # print("down_speed !! [Reward]", temp, " [dist]", round(dist,1))
        return reward

    def get_curveinfo(self, sensing_info):
        # ==========================================================#
        # Calculate speed reward for detecting curves 
        # ==========================================================#
        curve_info = []
        curve_position = 1
        SHARP_CURVE = 20
        change_angles = []
        for x in range(0, 9):
            diff_angle = sensing_info.track_forward_angles[x+1] - sensing_info.track_forward_angles[x]
            change_angles.append(abs(diff_angle))
            if abs(diff_angle) > SHARP_CURVE:
                if diff_angle > 0:
                    curve_position = -1

            # if x < 5 and change_rate > 20:
            #     up_speed = False

        curve_angle = max(change_angles)
        curve_index = change_angles.index(curve_angle)

        # # in the max curve area
        # DISTANCE_DECAY_RATE = -4.8
        if curve_angle > SHARP_CURVE and curve_index < 5:
            # baseline = ((math.exp(-(abs(curve_index))) * DISTANCE_DECAY_RATE) + 4) * curve_position       
            # print("[curve]", curve_index, ": baseline:", baseline)
        
            # curve_info.append(baseline)
            curve_info.append(curve_index)
            curve_info.append(curve_angle)

        return curve_info

    def compute_angle_reward(self, sensing_info):
        ANGLE_REWARD = 0
        TRACK_OUTLINE = 10
        DISTANCE_DECAY_RATE = 0.8
        dist_to_outline = abs(sensing_info.to_middle) - TRACK_OUTLINE
        
        RIGHT_AVOID_ANGLE = sensing_info.to_middle > 5 and sensing_info.moving_angle > 0
        LEFT_AVOID_ANGLE = sensing_info.to_middle < -5 and sensing_info.moving_angle < 0
        if RIGHT_AVOID_ANGLE or LEFT_AVOID_ANGLE:
            ANGLE_REWARD = math.exp(-(abs(dist_to_outline) * DISTANCE_DECAY_RATE))
        
        return (round(ANGLE_REWARD,1) * -1)

    def get_baseline(self, sensing_info):
        # =========================================================== #
        # Calculate new centerline when finding obstacles
        # =========================================================== #
        thresh_dist = self.half_road_limit
        baseline_info = []
        # DISTANCE_TO_AVOID = 2.3  # Do not use (Just use for printing)
        # DISTANCE_TO_MOVE = 2.5 # Do not use '2.5' too close to obastables (Bad Result)
        OBSTACLE_WIDTH = 2.3
        CAR_WIDTH = 2.5
        SLOPE_CONST = -50.0
        DELTA_Y = 0 - SLOPE_CONST
        DELTA_X = (OBSTACLE_WIDTH + CAR_WIDTH)/2
        SLOPE_LEFT = DELTA_Y / DELTA_X * -1        
        SLOPE_RIGHT = DELTA_Y / DELTA_X

        if len(sensing_info.track_forward_obstacles) > 0:
            o_dist, o_to_middle = sensing_info.track_forward_obstacles[0]
            if 0 < o_dist < 50:
                DISTANCE_TO_MOVE = (abs(o_to_middle) + thresh_dist) / 3 
                if o_to_middle < -1:
                    baseline = o_to_middle + DISTANCE_TO_MOVE
                elif o_to_middle > 1:
                    baseline = o_to_middle - DISTANCE_TO_MOVE
                else:
                    # Obstacles near the centerline move according to "my car position"
                    if sensing_info.to_middle >= 0:
                        baseline = o_to_middle + DISTANCE_TO_MOVE
                    else:
                        baseline = o_to_middle - DISTANCE_TO_MOVE
                      
                # Do not use 'thresh_left/thresh_right' for (Bad Result)
                # thresh_left = o_to_middle - DISTANCE_TO_AVOID
                # thresh_right = o_to_middle + DISTANCE_TO_AVOID
                thresh_left = ((o_dist*-1) - SLOPE_CONST) / SLOPE_LEFT + o_to_middle
                thresh_right = ((o_dist*-1) - SLOPE_CONST) / SLOPE_RIGHT + o_to_middle
                baseline_info.append(baseline)
                baseline_info.append(thresh_left)
                baseline_info.append(thresh_right)
                baseline_info.append(o_dist)

        return baseline_info
                    
    # =========================================================== #
    # Reward Function
    # =========================================================== #
    def compute_reward(self, sensing_info):
        # =========================================================== #
        # Area for writing code
        # =========================================================== #
        
        thresh_dist = self.half_road_limit  # 4 wheels off the track
        dist = abs(sensing_info.to_middle)
        DISTANCE_DECAY_RATE = 1.2        # The rate at which the reward decays for the distance function

        # [Obstacles]
        obstacle_info = self.get_baseline(sensing_info) 

        # [Speed]
        CURVE_REWARD = 0
        curve_info = self.get_curveinfo(sensing_info)
        if len(curve_info) > 0:
            curve_index = curve_info[0]
            curve_angle = curve_info[1]
            if curve_index == 1:
                if abs(sensing_info.moving_angle) - curve_angle < 5:
                    CURVE_REWARD = 0.5

        # [ANGLE]
        ANGLE_REWARD = self.compute_angle_reward(sensing_info)

        if dist > thresh_dist:
            reward = -1
        elif sensing_info.collided:
            reward = -1 
        elif len(obstacle_info) > 0:
            baseline = obstacle_info[0] 
            thresh_left = obstacle_info[1]
            thresh_right = obstacle_info[2]
            o_dist = obstacle_info[3]

            if sensing_info.to_middle > thresh_left and sensing_info.to_middle < thresh_right:
                reward = math.exp(-(abs(o_dist/10) * DISTANCE_DECAY_RATE)) * -1
            else:
                reward = math.exp(-(abs(sensing_info.to_middle - baseline) / 2 * DISTANCE_DECAY_RATE)) + ANGLE_REWARD + CURVE_REWARD
                
            reward = round(reward,2)
            print("[Obstacle] ", reward, "dist: ", round(sensing_info.to_middle, 2), ", [base]", round(baseline, 2), "steering:", sensing_info.moving_angle, "angle_reward:", ANGLE_REWARD, "curve_reward:", CURVE_REWARD)

        # elif len(curve_info) > 0:
        #     baseline = curve_info[0]
        #     curve_index = curve_info[1]
        #     curve_angle = curve_info[2]
        #     CURVE_REWARD = 0

        #     if curve_index == 0:
        #         if abs(sensing_info.moving_angle) - curve_angle < 5:
        #             CURVE_REWARD = 0.5

        #     reward = math.exp(-(abs(sensing_info.to_middle - baseline) / 2 * DISTANCE_DECAY_RATE)) + CURVE_REWARD + ANGLE_REWARD
        #     reward = round(reward,2)
        #     print("[Curve] ", reward, "dist: ", round(sensing_info.to_middle, 2), ", [base]", round(baseline, 2), "curve_index:", curve_index, "curve_angle:", curve_angle, "steering:", sensing_info.moving_angle)

        else:
            reward = math.exp(-(dist / 2 * DISTANCE_DECAY_RATE)) + ANGLE_REWARD + CURVE_REWARD

            reward = round(reward,2)
            ## print("[General] ", reward, ", dist: ", round(dist, 2), ", speed_reward:", CURVE_REWARD, "angle_reward:", ANGLE_REWARD)
            print("[General] ", reward, ", dist: ", round(dist, 2), ", angle_reward:", ANGLE_REWARD, "curve_reward:", CURVE_REWARD)
        
        #
        # Editing area ends
        # ==========================================================#
        return reward

    # =========================================================== #
    # Model network
    # =========================================================== #
    def build_custom_model(self):
        model = Sequential()
        # ?�이?? ?�기 (?�드?? 개수: 32, relu: ?�성?�수, he_uniform: Weight 초기�?)
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        # ?�이?? 구성 & 로그?�성
        model.summary()
        
        # Loss ?�수 (learning_rate 지?? 가??)
        model.compile(loss='mse', optimizer=Adam(lr=self.dqn_param.learning_rate))

        return model


if __name__ == "__main__":
    client = DQNCustomClient()

    client.override_model()

    if model_load:
        client.agent.load_model(model_weight_path)

    client.run(training_duration)
    sys.exit()
