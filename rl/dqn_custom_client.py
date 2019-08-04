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

# Try 1
# model_weight_path = "./save_model/dqn_weight_T0731_184922_speedmap.h5" 
# episode: 2011   score: 27.7 [score] 351.5 / 50.54 % (= 7.0 ), episode: 757  vs.   [mean] 128.3 / 11.83 % (= 10.8 ), episode: 1321 

# Try 2
# model_weight_path = "./save_model/dqn_weight_T0803_213413_200_maxscore_pass_finishline.h5"
# episode: 251  score: 110.9  check point reached: 17  lap: 4.84 [score] episode: 129 178.5 / 4.57 % (= 39.1 )

# Try 3
# model_weight_path = "./save_model/dqn_weight_T0804_004604_130_throttle_test.h5"
# episode: 1146  score: 135.2  check point reached: 27  lap: 7.26 [score]  1425.9 / 100.0 % (= 14.3 ), episode: 759

# Try 4
# model_weight_path = "./save_model/dqn_weight_T0804_013106_760_throttle_test_pass_finishline.h5"
# episode: 321  score: 83.9  check point reached: 17  lap: 4.84 [score]  1214.3 / 80.38 % (= 15.1 ), episode: 300

# Try 5
# model_weight_path = "./save_model/dqn_weight_T0804_082415_300_throttle_test_p80.h5"
# episode: 715  score: 193.1  check point reached: 39  lap: 10.75 [score]  2260.8 / 100.0 % (= 22.6 ), episode: 419

# ----- ignore -----
## Try 6
## model_weight_path = "./save_model/dqn_weight_T0804_094245_420_throttle_test_pass_finishline.h5"
## episode: 784  score: 62.6  check point reached: 14  lap: 4.03 [score]  1969.3 / 100.0 % (= 19.7 ), episode: 629

## Try 7
## model_weight_path = "./save_model/dqn_weight_T0804_150602_630_throttle_test_pass_finishline.h5"
## episode: 526  score: 74.9  check point reached: 18  lap: 5.11 [score]  526.2 / 26.08 % (= 20.2 ), episode: 461

# ----- again -----
# Try 6
# model_weight_path = "./save_model/dqn_weight_T0804_094245_420_throttle_test_pass_finishline.h5"
# episode: 170  score: 284.5  check point reached: 44  lap: 12.1 [score]  2579.9 / 100.0 % (= 25.8 ), episode: 144

# Try 7
model_weight_path = "./save_model/dqn_weight_T0805_085758_150_throttle_test_pass_finishline.h5"

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
            dict(throttle=0.9, steering=0)
        ]
        #
        # Editing area ends
        # ==========================================================#
        return actions

    # =========================================================== #
    # Reward Function
    # =========================================================== #
    def compute_reward(self, sensing_info):

        # =========================================================== #
        # Area for writing code
        # =========================================================== #
        # Editing area starts from here
        #
        thresh_dist = self.half_road_limit  # 4 wheels off the track
        dist = abs(sensing_info.to_middle)
        avoid_o_to_middle = 10
        up_speed = True
        up_speed_reward = 0
        down_speed_reward = 0

        # sensing_info:
        # sensing_info.collided
        # sensing_info.speed
        # sensing_info.moving_forward
        # sensing_info.moving_angle
        # sensing_info.lap_progress
        # sensing_info.track_forward_angles
        # sensing_info.track_forward_obstacles

        # ?�애물을 발견?? 경우, 중앙?�로부?�의 거리 차�? ?�수�? 보상?? ?�다
        if len(sensing_info.track_forward_obstacles) > 0:
            o_dist, o_to_middle = sensing_info.track_forward_obstacles[0]
            if o_dist < 50:
                avoid_o_to_middle = abs(sensing_info.to_middle - o_to_middle)

        # ?�방 주행각도 변?�량 ?�보
        change_rate_angles = []
        for x in range(0, 9):
            change_rate = abs(sensing_info.track_forward_angles[x+1] - sensing_info.track_forward_angles[x])
            change_rate_angles.append(change_rate)
            # if x < 3 and change_rate > 15:
            if x < 3 and change_rate > 20:
                up_speed = False

        max_change_value = max(change_rate_angles)
        max_change_index = change_rate_angles.index(max_change_value)

        # 커브각도가 15 ?�상?? 코너�? 구간?? 근접?? 경우
        if max_change_value > 15:
            # if max_change_index < 4 and max_change_index > 0:
            if max_change_index < 3 and max_change_index > 0:
                up_speed = False

        if up_speed == True and sensing_info.speed > 40:
            up_speed_reward = 0.2
        elif up_speed == False and sensing_info.speed < 30:
            down_speed_reward = 0.2
                            
        # ?�랙?? 각도?� 차량?? 각도 차이가 ?�을?�록 보상?? ?�다
        # if len(sensing_info.track_forward_angles) > 0:
        #     diff_angles = abs(sensing_info.track_forward_angles - sensing_info.moving_angles)

        if dist > thresh_dist:
            reward = -1
        elif sensing_info.collided:
            reward = -1
        elif avoid_o_to_middle < 2.5:
            reward = -0.5   # -1 �? 주면 frozen ?�인?�듯
        # elif max_change_value < 15 and sensing_info.speed > 40:
        #     reward = 0.8        
        else:
            if dist > 5:
                reward = 0.1
            elif dist > 4:
                reward = 0.2
            elif dist > 3:
                reward = 0.4 + up_speed_reward
            elif dist > 2:
                reward = 0.6 + up_speed_reward
            elif dist > 1:
                reward = 0.8 + down_speed_reward + up_speed_reward
            else:
                reward = 1 + down_speed_reward + up_speed_reward

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
