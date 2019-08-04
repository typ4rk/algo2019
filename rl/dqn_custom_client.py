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

# model_weight_path = "./save_model/dqn_weight_T0804_094245_420_throttle_test_pass_finishline.h5"
# episode: 784  score: 62.6  check point reached: 14  lap: 4.03 [score]  1969.3 / 100.0 % (= 19.7 ), episode: 629

model_weight_path = "./save_model/dqn_weight_T0804_150602_630_throttle_test_pass_finishline.h5"
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
        weight_dist_0 = 1
        weight_dist_1 = 0.8
        weight_dist_2 = 0.6
        weight_dist_3 = 0.4
        weight_dist_4 = 0.2 
        weight_dist_5 = 0.1
        
        # sensing_info:
        # sensing_info.collided
        # sensing_info.speed
        # sensing_info.moving_forward
        # sensing_info.moving_angle
        # sensing_info.lap_progress
        # sensing_info.track_forward_angles
        # sensing_info.track_forward_obstacles

        # 장애물을 발견한 경우, 중앙으로부터의 거리 차가 클수록 보상이 높다
        if len(sensing_info.track_forward_obstacles) > 0:
            o_dist, o_to_middle = sensing_info.track_forward_obstacles[0]
            if o_dist < 50:
                avoid_o_to_middle = abs(sensing_info.to_middle - o_to_middle)

        # 전방 주행각도 변화량 정보
        change_rate_angles = []
        for x in range(0, 9):
            change_rate_angles.append(abs(sensing_info.track_forward_angles[x+1] - sensing_info.track_forward_angles[x]))

        max_change_value = max(change_rate_angles)
        max_change_index = change_rate_angles.index(max_change_value)

        # 커브각도가 15 이상인 코너링 구간에 근접한 경우
        if max_change_value > 15:
            # go inside!! (make: 1)
            if max_change_index < 4 and max_change_index > 0:
                # weight_dist_1 = 0.9
                # weight_dist_2 = 0.2
                # weight_dist_3 = 0.2
                # weight_dist_4 = 0.1
                # weight_dist_5 = 0.1
                if sensing_info.speed < 30:
                    weight_dist_1 = 0.9
                    weight_dist_2 = 0.2
                    weight_dist_3 = 0.2
                    weight_dist_4 = 0.1
                    weight_dist_5 = 0.1
                else:
                    weight_dist_1 = 0.9
                    weight_dist_2 = 0.2
                    weight_dist_3 = 0.2
                    weight_dist_4 = 0.1
                    weight_dist_5 = 0.1
            # go outside!! (make: 0.6)           
            # else:
            #     weight_dist_4 = 0.4
            #     weight_dist_5 = -0.1
                            
        # 트랙의 각도와 차량의 각도 차이가 작을수록 보상이 높다
        # if len(sensing_info.track_forward_angles) > 0:
        #     diff_angles = abs(sensing_info.track_forward_angles - sensing_info.moving_angles)

        if dist > thresh_dist:
            reward = -1
        elif sensing_info.collided:
            reward = -1
        elif avoid_o_to_middle < 2.5:
            reward = -0.5   # -1 로 주면 frozen 원인인듯
        elif max_change_value < 15 and sensing_info.speed > 40:
            reward = 0.8
        else:
            if dist > 5:
                reward = weight_dist_5
            elif dist > 4:
                reward = weight_dist_4
            elif dist > 3:
                reward = weight_dist_3
            elif dist > 2:
                reward = weight_dist_2
            elif dist > 1:
                reward = weight_dist_1
            else:
                reward = weight_dist_0

            
        #
        # Editing area ends
        # ==========================================================#
        return reward

    # =========================================================== #
    # Model network
    # =========================================================== #
    def build_custom_model(self):
        model = Sequential()
        # 레이어 쌓기 (노드의 개수: 32, relu: 활성함수, he_uniform: Weight 초기값)
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        # 레이어 구성 & 로그작성
        model.summary()
        
        # Loss 함수 (learning_rate 지정 가능)
        model.compile(loss='mse', optimizer=Adam(lr=self.dqn_param.learning_rate))

        return model


if __name__ == "__main__":
    client = DQNCustomClient()

    client.override_model()

    if model_load:
        client.agent.load_model(model_weight_path)

    client.run(training_duration)
    sys.exit()
