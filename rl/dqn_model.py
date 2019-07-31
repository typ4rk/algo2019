import setup_path
import airsim
import os
import time
import math
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from airsim_env import AirSimEnv
from abc import abstractmethod

# =========================================================== #
# Global Configurations
# =========================================================== #
enable_api_control = True  # True(Api Control) /False(Key board control)
is_debug = False
current_clock_speed = 1

# =========================================================== #


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, dqn_param):

        # ?�태?� ?�동?? ?�기 ?�의
        self.state_size = state_size
        self.action_size = action_size

        # DQN ?�이?�파?��???
        self.discount_factor = dqn_param.discount_factor
        self.learning_rate = dqn_param.learning_rate
        self.epsilon_decay = dqn_param.epsilon_decay
        self.epsilon_min = dqn_param.epsilon_min
        self.epsilon = dqn_param.epsilon
        self.batch_size = dqn_param.batch_size
        self.train_start = dqn_param.train_start
        # 리플?�이 메모�?, 최�? ?�기 20000
        self.memory = deque(maxlen=dqn_param.memory_size)

        self.model = self.build_model()
        self.target_model = self.build_model()

        # ?��? 모델 초기??
        self.update_target_model()

    def load_model(self, file_path):
        if not os.path.exists(file_path):
            raise IncorrectAction("Weight file not found!!")
        else:
            self.model.load_weights(file_path)

    # ?�태가 ?�력, ?�함?��? 출력?? ?�공?�경�? ?�성
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    # ?��? 모델?? 모델?? 가중치�? ?�데?�트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # ?�실�? ?�욕 ?�책?�로 ?�동 ?�택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def get_eval_action(self, state):
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    # ?�플 <s, a, r, s'>?? 리플?�이 메모리에 ?�??
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플?�이 메모리에?? 무작?�로 추출?? 배치�? 모델 ?�습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에?? 배치 ?�기만큼 무작?�로 ?�플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # ?�재 ?�태?? ?�?? 모델?? ?�함??
        # ?�음 ?�태?? ?�?? ?��? 모델?? ?�함??
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정?�을 ?�용?? ?�데?�트 ?��?
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


# DQN Hyper param
class DQNParam:
    # default values
    discount_factor = 0.99
    learning_rate = 0.00025
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    batch_size = 100
    train_start = 1000
    # replay memory size : max 20000
    memory_size = 20000


class CarState:
    def __init__(self, name):
        self.__name = name

    collided = False
    collision_distance = 0
    speed = 0
    to_middle = 0
    moving_angle = 0

    moving_forward = True
    lap_progress = 0
    track_forward_angles = []
    track_forward_obstacles = []
    distance_to_way_points = []


class DQNClient:

    def __init__(self, dqn_param):
        self.player_name = ""

        self.car_controls = airsim.CarControls()

        self.client = airsim.CarClient()
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(enable_api_control, self.player_name)

        self.airsim_env = AirSimEnv()
        self.way_points, self.obstacle_points = self.airsim_env.load_track_info(self.client)
        self.collision_time_stamp = 0
        self.sensing_info = CarState(self.player_name)
        self.all_obstacles = self.airsim_env.get_all_obstacle_info(self.obstacle_points, self.way_points)
        self.state_size = self.airsim_env.get_state_size()
        self.frozen_count = 0
        self.car_current_pos_x, self.car_next_pos_x = 0, 0

        # road half width + car half width
        self.half_road_limit = self.client.getAlgoUserAPI().ac_road_width_half + 1.25

        self.control_interval = round(0.1 / current_clock_speed,2)

        if len(self.action_space()) < 1:
            raise IncorrectAction("Please check the action definition : At least one action is required")

        self.action_size = len(self.action_space())

        # DQN ?�이?�트 ?�성
        self.agent = DQNAgent(self.state_size, self.action_size, dqn_param)

        # running client id �? ?�더 ?�성
        now = time.localtime()
        self.run_cid = "T%02d%02d_%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        # ?�간 ?�성.
        self.start_time = time.time()
        self.end_time = 0


    @staticmethod
    def make_initial_movement(car_controls, client):
        # 조금 주행?? ?�킨??.
        car_controls.throttle = 1
        car_controls.steering = 0
        client.setCarControls(car_controls)
        time.sleep(round(2/current_clock_speed,2))

    def calc_sensing_data(self, car_next_state, car_current_state, backed_car_state, way_points, check_point_index, progress):
        distance_from_center = self.airsim_env.get_distance_from_center(car_next_state, way_points,
                                                                        check_point_index)
        right_of_center = self.airsim_env.is_right_of_center(car_next_state, way_points, check_point_index)
        self.sensing_info.to_middle = distance_from_center * (1 if right_of_center else -1)
        self.sensing_info.speed = self.airsim_env.get_speed(car_next_state)
        self.sensing_info.moving_forward = self.airsim_env.is_moving_forward(car_current_state, car_next_state,
                                                                             way_points,
                                                                             check_point_index)
        # ?��??? ?�는 ?�태?�서 각도�? 구할 ?? ?�으므�?, 좌표가 ?�랐?? 마�?�? ?�태�? 기억?�여 ?�다.
        self.car_current_pos_x = car_current_state.kinematics_estimated.position.x_val
        self.car_next_pos_x = car_next_state.kinematics_estimated.position.x_val
        if self.car_current_pos_x == self.car_next_pos_x:
            self.sensing_info.moving_angle = self.airsim_env.get_moving_angle(backed_car_state, car_next_state,
                                                                              way_points,
                                                                              check_point_index)
        else:
            self.sensing_info.moving_angle = self.airsim_env.get_moving_angle(car_current_state, car_next_state,
                                                                              self.way_points,
                                                                              check_point_index)
        collision_info = self.client.simGetCollisionInfo(self.player_name)
        if collision_info.has_collided:
            if self.collision_time_stamp < collision_info.time_stamp:
                self.sensing_info.collided = True
            else:
                self.sensing_info.collided = False
        else:
            self.sensing_info.collided = False
        self.collision_time_stamp = collision_info.time_stamp

        self.sensing_info.lap_progress = progress
        self.sensing_info.track_forward_angles = self.airsim_env.get_track_forward_angle(car_next_state,
                                                                                         self.way_points,
                                                                                         check_point_index)
        self.sensing_info.track_forward_obstacles = self.airsim_env.get_track_forward_obstacle(car_next_state,
                                                                                               self.way_points,
                                                                                               check_point_index,
                                                                                               self.all_obstacles)
        self.sensing_info.distance_to_way_points = self.airsim_env.get_distance_to_way_points(car_next_state,
                                                                                              self.way_points,
                                                                                              check_point_index)
        return self.sensing_info

    def run(self, time_limit_hour):

        os.makedirs("./save_model/" + str(self.run_cid))
        os.makedirs("./save_graph/" + str(self.run_cid))

        car_prev_state = self.client.getCarState(self.player_name)
        # 조금 주행?? ?�킨??.
        self.make_initial_movement(self.car_controls, self.client)

        check_point_index = 0
        car_current_state = self.client.getCarState(self.player_name)
        backed_car_state = car_current_state

        scores, episodes = [], []
        current_episode = 0
        scores_per_episode = []
        frozen = 0
        max_score = 0
        save_episode = 0
        save_lap_progress = 0
        
        # print("agent_current_state:{}".format(car_current_state))
        cur_lab = 1
        half_complete_flag = False
        finish = False
        time_limit_sec = time_limit_hour * 60 * 60
        # while 루프?�작.
        while not finish:
            # ?�재 ?�태 구성
            agent_current_state = self.airsim_env.get_current_state(car_current_state, car_prev_state, self.way_points,
                                                                    check_point_index, self.all_obstacles)
            # print(agent_current_state)
            agent_current_state = np.reshape(agent_current_state, [1, self.state_size])
            check_point_index, _ = self.airsim_env.get_current_way_points(car_current_state, self.way_points,
                                                                          check_point_index)

            # ?��??�이?�에 ?�어�? ?�는??(# ?�택?? ?�동?�로 ?�경?�서 ?? ?�?�스?? 진행)
            action = self.agent.get_action(agent_current_state)
            self.car_controls = self.interpret_action(action, self.car_controls)
            self.client.setCarControls(self.car_controls)
            time.sleep(self.control_interval)

            # ?�음 ?�태
            car_next_state = self.client.getCarState(self.player_name)
            check_point_index, _ = self.airsim_env.get_current_way_points(car_next_state, self.way_points,
                                                                          check_point_index)
            agent_next_state = self.airsim_env.get_current_state(car_next_state, car_current_state, self.way_points,
                                                                 check_point_index, self.all_obstacles)
            agent_next_state = np.reshape(agent_next_state, [1, self.state_size])

            progress = self.airsim_env.get_progress(car_current_state, self.way_points, check_point_index, cur_lab)
            # print("progress", progress)
            if progress >= 52:
                half_complete_flag = False
            elif progress >= 50:
                half_complete_flag = True
            if half_complete_flag and round(progress) == 0:
                cur_lab = 2
                progress = 50.0 + progress

            # ?�싱 ?�이?? 계산
            sensing_info = self.calc_sensing_data(car_next_state, car_current_state, backed_car_state, self.way_points,
                                                  check_point_index, progress)

            # 보상 ?�수�? ?�라미터�? ?�겨준??.
            reward = self.compute_reward(sensing_info)


            # ?�기??  done ?� 보통?� ?�로 ?�하�? ?�탈?�서 ?�이?? 진행?�기 ?�려?? 경우.
            # frozen ?��??�이?��? ?�답 ?�는 경우. 리셋.
            done, frozen = self.is_done(car_next_state, car_current_state, reward, progress, frozen)

            if progress >= 100:
                done = 1

            scores_per_episode.append(reward)

            if is_debug:
                print("### cur_state", agent_current_state, ",action:", action, ",reward:", reward, ",next_stat:",
                      agent_next_state, done)

            # 리플?�이 메모리에 ?�플 <s, a, r, s'> ?�??
            self.agent.append_sample(agent_current_state, action, reward, agent_next_state, done)

            # �? ?�?�스?�마?? ?�습
            if len(self.agent.memory) >= self.agent.train_start:
                self.agent.train_model()

            if done:

                # ?? ?�피?�드가 ?�남.
                score = np.sum(scores_per_episode)
                episodes.append(current_episode)

                scores.append(round(score, 2))

                # 추이�? 보기 ?�해??
                graph_x_width = 500
                post_fix = math.floor((len(episodes) - 1) / graph_x_width)
                graph_start = post_fix * graph_x_width
                pylab.plot(episodes[graph_start:], scores[graph_start:], 'b')
                pylab.savefig("./save_graph/" + str(self.run_cid) + "/dqn_graph_" + str(post_fix) + ".png")
                if len(episodes) % graph_x_width == 0:
                    pylab.clf()

                if score > max_score:
                    max_score = score
                    save_episode = current_episode
                    save_lap_progress = sensing_info.lap_progress # check_point_index

                print("Num of steps done :", current_episode, "episode:", current_episode, "  score:", score,
                      " (max:", round(max_score,1), "/ episode:", save_episode, "/ lap_progress:", save_lap_progress, "%)",
                      "  memory length:",
                      len(self.agent.memory), "  epsilon:", self.agent.epsilon, " check point reached:",
                      check_point_index)

                if current_episode % 10 == 0:
                    self.agent.model.save_weights(
                        "./save_model/" + str(self.run_cid) + "/dqn_weight_" + str(current_episode) + ".h5")
                
                # 모델 ?�데?�트
                self.agent.update_target_model()

                self.client.reset()
                time.sleep(0.2)
                # 리셋?? 조금 주행?? ?�킨??.
                self.make_initial_movement(self.car_controls, self.client)
                # 변?�들 초기??
                car_next_state = self.client.getCarState(self.player_name)
                backed_car_state = self.client.getCarState(self.player_name)
                check_point_index = 0
                scores_per_episode = []
                cur_lab = 1
                half_complete_flag = False
                current_episode += 1

            if round(self.car_current_pos_x, 4) != round(self.car_next_pos_x, 4):
                backed_car_state = car_current_state
            car_prev_state = car_current_state
            car_current_state = car_next_state

            if time_limit_sec != 0 and time.time() - self.start_time > time_limit_sec:
                finish = True
            ##END OF LOOP

    def is_done(self, car_state, prev_car_state, reward, progress, frozen=0):
        done = 0
        if reward <= -1:
            done = 1
        elif progress > 2 and car_state.speed <= 1:
            done = 1
        elif car_state.kinematics_estimated.position.x_val == prev_car_state.kinematics_estimated.position.x_val and car_state.kinematics_estimated.position.y_val == prev_car_state.kinematics_estimated.position.y_val:
            frozen = frozen + 1
            if frozen > 10:
                frozen = 0
                done = 1
                print("Simulator frozen for some reason ==> Call, done!(reset)")
        return done, frozen

    def interpret_action(self, action, car_controls):
        selected_action = self.action_space()[action]
        car_controls.steering = selected_action['steering']
        car_controls.throttle = selected_action['throttle']
        return car_controls

    def override_model(self):
        self.agent.model = self.build_custom_model()
        self.agent.target_model = self.build_custom_model()
        # ?��? 모델 초기??
        self.agent.update_target_model()

    @abstractmethod
    def action_space(self):
        raise NotImplementedError('Implement me in subclass')

    @abstractmethod
    def compute_reward(self, sensing_info):
        raise NotImplementedError('Implement me in subclass')

    @abstractmethod
    def build_custom_model(self):
        raise NotImplementedError('Implement me in subclass')

class IncorrectAction(Exception):
    pass
