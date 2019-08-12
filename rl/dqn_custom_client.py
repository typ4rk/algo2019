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

# Try 8
# model_weight_path = "./save_model/dqn_weight_T0805_131733_90_throttle_test_pass_finishline.h5"
# map: SpeedRacing
# episode: 1870  score: 7.2  check point reached: 9  lap: 2.69 [score]  189.3 / 25.81 % (= 7.3 ), episode: 1103

# --------------------------------------------------------
# map: RL_Map1_block (이때만 잠깐 맵 변경해서 테스트)
#
# Try 1
# episode: 357  score: 450.0  check point reached: 71  lap: 27.91 [score]  1248.2 / 77.52 % (= 16.1 ), episode: 277
# --------------------------------------------------------

# map: SpeedRacing 복귀
# Try 2
# model_weight_path = "./save_model/dqn_weight_T0808_085342_280_RL_Map1_block_p77.h5"
# episode: 264  score: 108.8  check point reached: 14  lap: 4.03 [score]  321.7 / 10.22 % (= 31.5 ), episode: 188

# Try 3
# model_weight_path = "./save_model/dqn_weight_T0808_143501_190_p10.h5"
# episode: 101  score: 112.2  check point reached: 16  lap: 4.57 [score]  2043.2 / 76.08 % (= 26.9 ), episode: 100

# Try 4
# model_weight_path = "./save_model/dqn_weight_T0808_161332_100_p76.h5"
# episode: 604  score: 30.4  check point reached: 16  lap: 4.57 [score]  72.8 / 12.1 % (= 6.0 ), episode: 559

# Try 5
# model_weight_path = "./save_model/dqn_weight_T0808_202811_560_p12.h5"
# episode: 2151  score: 23.2  check point reached: 16  lap: 4.57 [score]  43.1 / 11.02 % (= 3.9 ), episode: 276

# Try 5
# model_weight_path = "./save_model/dqn_weight_T0808_223505_280_p11.h5"
# [NOT-SAVED]episode: 77  score: 3081.9  check point reached: 185  lap: 100.0 [score]  3081.9 / 100.0 % (= 30.8 ), episode: 77
# episode: 97  score: 752.0  check point reached: 96  lap: 26.08 [score]  3323.3 / 100.0 % (= 33.2 ), episode: 75

# [MAP] mariana
model_weight_path = "./save_model/dqn_weight_T0812_094057_80_pass_finishline.h5"

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
            if max_change_index < 4 and max_change_index > 0:
            # if max_change_index < 3 and max_change_index > 0:
                up_speed = False

        if up_speed == True:
            if sensing_info.speed > 40:
                up_speed_reward = 0.1

            # print("up_speed !! [Reward]", up_speed_reward, " [Speed]", sensing_info.speed)
        elif up_speed == False: # and sensing_info.speed < 30:
            down_speed_reward = 0.1
            if dist > 2:
                temp = 0
            elif dist > 1:
                temp = down_speed_reward
            else:
                temp = down_speed_reward * 2

            # print("down_speed !! [Reward]", temp, " [Speed]", sensing_info.speed)
            print("down_speed !! [Reward]", temp, " [dist]", round(dist,1))

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
                reward = 1 + down_speed_reward*2 + up_speed_reward*2

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
