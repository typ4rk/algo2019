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
#model_weight_path = "./save_model/T0820_145117/dqn_weight_29.h5"   -- base 
# model_weight_path = "./save_model/T0821_095102/dqn_weight_10.h5"
# model_weight_path = "./save_model/dqn_weight_1758.h5"   # marina best model 
# model_weight_path = "./save_model/T0821_185205/dqn_weight_0.h5"    # speed 완주모델 
# model_weight_path = "./save_model/T0821_204655/dqn_weight_1.h5"    # speed 완주모델 Best

# model_weight_path = "./save_model/dqn_weight_150.h5"    # obum 
# model_weight_path = "./save_model/T0822_083125/dqn_weight_16.h5"    # 마리나 183
model_weight_path = "./dqn_weight_150.h5"    # 마리나 178

# model_weight_path = "./save_model/T0822_103413/dqn_weight_4.h5"

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
        dqn_param.learning_rate = 0.0
        dqn_param.epsilon = 0.0
        dqn_param.epsilon_decay = 0.9999
        dqn_param.epsilon_min = 0.01
        dqn_param.batch_size = 100
        dqn_param.train_start = 2000
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
            dict(throttle=1.00, steering=0.05),            
            dict(throttle=1.00, steering=-0.05),                             
            
            dict(throttle=0.93, steering=0.20),              # 85  20    -   88  20
            dict(throttle=0.93, steering=-0.20),
                                       
            dict(throttle=0.55, steering=0.55),              # 71  60    -   78  55
            dict(throttle=0.55, steering=-0.55),
            
            dict(throttle=0.00, steering=0),
            dict(throttle=1.00, steering=0)
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
        
        reward = 0.0
        
        thresh_dist = self.half_road_limit  # 4 wheels off the track
        
        # 도로 중앙에서의 거리
        dist = abs(sensing_info.to_middle)

        # 도로를 벗어나거나 충돌의 경우 종료
        if dist > (thresh_dist+1):
            return -1.0
        elif sensing_info.collided:
            return -100.0
       
        # 차량 속도 보상
        speed = sensing_info.speed
        
        # 완주보상
        complete = sensing_info.lap_progress

        reward += (speed / 10.0) ** 2.0 if speed > 75.0 else 0.0

        if complete > 99:   reward += 10000.0
        elif complete > 90:  reward += 10.0
        elif complete > 80:  reward += 5.0
        elif complete > 70:  reward += 1.0
        elif complete > 60:  reward += 0.5  
        elif complete > 50:  reward += 0.1  
        
        # Editing area ends
        # ==========================================================#
        return reward

    # =========================================================== #
    # Model network
    # =========================================================== #
    def build_custom_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu',
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