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
# episode: 2959  score: 24.0  check point reached: 15  lap: 4.3 [score]  125.0 / 39.25 % (= 3.2 ), episode: 2487
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

    def get_baseline(self, sensing_info):
        # =========================================================== #
        # Calculate new centerline when finding obstacles
        # =========================================================== #
        thresh_dist = self.half_road_limit
        baseline_info = []
        DISTANCE_TO_AVOID = 2.3  # Do not use (Just use for printing)
        # DISTANCE_TO_MOVE = 2.5 # Do not use '2.5' too close to obastables (Bad Result)

        if len(sensing_info.track_forward_obstacles) > 0:
            o_dist, o_to_middle = sensing_info.track_forward_obstacles[0]
            if o_dist < 50:
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
                thresh_left = o_to_middle - DISTANCE_TO_AVOID
                thresh_right = o_to_middle + DISTANCE_TO_AVOID                
                baseline_info.append(baseline)
                baseline_info.append(thresh_left)
                baseline_info.append(thresh_right)

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
        baseline_info = self.get_baseline(sensing_info) 

        # [Speed]
        CENTER_SPEED_REWARD = self.compute_speed_reward(sensing_info)

        if dist > thresh_dist:
            reward = -1
        elif sensing_info.collided:
            reward = -1
        elif len(baseline_info) > 0:
            baseline = baseline_info[0] 
            # thresh_left = baseline_info[1]
            # thresh_right = baseline_info[2]

            # if sensing_info.to_middle > thresh_left and sensing_info.to_middle < thresh_right:
            #     reward = 0
            # else:
            reward = math.exp(-(abs(sensing_info.to_middle - baseline) * DISTANCE_DECAY_RATE))
                # print("baseline:", abs(sensing_info.to_middle - baseline), "calc:", abs(sensing_info.to_middle - baseline) * DISTANCE_DECAY_RATE)
            
            print("[Reward] ", round(reward,3), "dist: ", round(sensing_info.to_middle, 2), ", [base]", round(baseline, 2), "L:", round(thresh_left,2), "R:", round(thresh_right, 2))
        else:
            reward = math.exp(-(dist * DISTANCE_DECAY_RATE)) + CENTER_SPEED_REWARD
            print("[Reward] ", round(reward,3), ", dist: ", round(dist, 2), ", down_speed_reward:", CENTER_SPEED_REWARD)

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
