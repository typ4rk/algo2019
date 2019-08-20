import setup_path
import time
import numpy as np
import math
import airsim
from airsim_env import AirSimEnv

# =========================================================== #
# Global Configurations
##
enable_api_control = False  # True(Api Control) /False(Key board control)
is_debug = True
# =========================================================== #

class DQNRewardTester():
    # =========================================================== #
    # Reward function to test
    # =========================================================== #
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

        # first_curve_index = -1
        thresh_angle = 20

        i = 1
        while i < len(tfa):
            diff_angle = abs(tfa[i] - tfa[i - 1])
            tfa_differences.append(diff_angle)
            # if first_curve_index = -1 and diff_angle > thresh_angle:
            #     first_curve_index = i
            i = i + 1

        max_diff_angle = max(tfa_differences)
        max_angle_dist = tfa_differences.index(max_diff_angle)        
        max_angle = tfa[max_angle_dist]

        i = max_angle_dist
        if tfa[i + 1] - tfa[i] < 0:
            max_angle = max_angle * -1

        if max_diff_angle < 10:
            reward_value = round(math.exp(-max(abs(ma) - 5, 0)), 2)
            print("= reward:", reward_value, "max_diff_angle:", max_diff_angle)
        else:
            if max_angle_dist == 0:
                reward_value = 1.0
            elif 1 <= max_angle_dist < 3:
                reward_value = 1.0 - min(abs(max_angle - ma), thresh_angle)/thresh_angle
                # reward_value = 1.0 - abs(max_angle - ma)/thresh_angle
                print("= reward:", reward_value, ", max_angle:", max_angle, "diff:", abs(max_angle-ma))
            else:
                reward_value = 1.0 - min(abs(tfa[0] - ma), thresh_angle)/thresh_angle
                # reward_value = 1.0 - abs(tfa[0] - ma)/thresh_angle
                print("= reward:", reward_value)

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

        # if 0.5 < dist_reward_value:
        angle_reward_value = self.calc_angle_reward_value(sensing_info)

        # if 1.0 < dist_reward_value + angle_reward_value:
        #     speed_reward_value = self.calc_speed_reward_value(sensing_info)

        reward = speed_reward_value + dist_reward_value + angle_reward_value + fc + thresh_reward_value

        # print(f"[Reward]{reward:0.3f} [to_middle]{round(self.sensing_info.to_middle,2)}, D:{dist_reward_value:0.3f} [angle]track:{self.sensing_info.track_forward_angles[0]} A:{angle_reward_value:0.3f} T:{thresh_reward_value:0.3f} [etc]S:{speed_reward_value:0.3f}")
        print(f"[Reward]{reward:0.3f} [to_middle]{round(self.sensing_info.to_middle,2)}, D:{dist_reward_value:0.3f} \
            [angle]track:{self.sensing_info.track_forward_angles[0]} angle:{self.sensing_info.moving_angle} A:{angle_reward_value:0.3f} T:{thresh_reward_value:0.3f} \
            [etc]speed:{self.sensing_info.speed} S:{speed_reward_value:0.3f}")        
        #
        # Editing area ends
        # ==========================================================#
        return reward


    def __init__(self):
        self.player_name = ""
        self.check_point_index = 0
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
        # road half width + car half width
        self.half_road_limit = self.client.getAlgoUserAPI().ac_road_width_half + 1.25


    def make_initial_movement(self, car_controls, client):
        car_controls.throttle = 1
        car_controls.steering = 0
        client.setCarControls(car_controls)
        time.sleep(1)

    def run(self):

        car_prev_state = self.client.getCarState(self.player_name)
        # 조금 주행을 시킨다.
        self.make_initial_movement(self.car_controls, self.client)
        car_current_state = self.client.getCarState(self.player_name)
        backed_car_state = car_current_state

        check_point_index = 0

        # while 루프시작.
        while True:
            # 현재 상태 구성
            car_current_state = self.client.getCarState(self.player_name)

            check_point_index, _ = self.airsim_env.get_current_way_points(car_current_state, self.way_points,
                                                                          check_point_index)

            # 센싱 데이터 계산
            sensing_info = self.calc_sensing_data(car_current_state, car_prev_state, backed_car_state,
                                                  self.way_points,
                                                  check_point_index)

            agent_current_state = self.airsim_env.get_current_state(car_current_state, car_prev_state, self.way_points,
                                                                    check_point_index, self.all_obstacles)
            # print(agent_current_state)
            # 보상 함수로 파라미터를 넘겨준다.
            reward = self.compute_reward(sensing_info)
            # print("Reward value : {}".format(reward))

            if round(self.car_current_pos_x, 4) != round(self.car_next_pos_x, 4):
                backed_car_state = car_current_state
            car_prev_state = car_current_state

            # if is_debug:
            #     print("=========================================================")
            #     print("to middle: {}".format(sensing_info.to_middle))

            #     print("collided: {}".format(sensing_info.collided))
            #     print("car speed: {} km/h".format(sensing_info.speed))

            #     print("is moving forward: {}".format(sensing_info.moving_forward))
            #     print("moving angle: {}".format(sensing_info.moving_angle))
            #     print("lap_progress: {}".format(sensing_info.lap_progress))

            #     print("track_forward_angles: {}".format(sensing_info.track_forward_angles))
            #     print("track_forward_obstacles: {}".format(sensing_info.track_forward_obstacles))
            #     print("distance_to_way_points: {}".format(sensing_info.distance_to_way_points))
            #     print("=========================================================")

            time.sleep(0.5)
            ##END OF LOOP

    def calc_sensing_data(self, car_next_state, car_current_state, backed_car_state, way_points, check_point_index):
        distance_from_center = self.airsim_env.get_distance_from_center(car_next_state, way_points,
                                                                        check_point_index)
        right_of_center = self.airsim_env.is_right_of_center(car_next_state, way_points, check_point_index)
        self.sensing_info.to_middle = distance_from_center * (1 if right_of_center else -1)
        self.sensing_info.speed = self.airsim_env.get_speed(car_next_state)
        self.sensing_info.moving_forward = self.airsim_env.is_moving_forward(car_current_state, car_next_state,
                                                                             way_points,
                                                                             check_point_index)
        # 정지해 있는 상태에서 각도를 구할 수 없으므로, 좌표가 달랐던 마지막 상태를 기억하여 둔다.
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

        self.sensing_info.lap_progress = self.airsim_env.get_progress(car_next_state, self.way_points,
                                                                      check_point_index, 1, 1)
        self.sensing_info.track_forward_angles = self.airsim_env.get_track_forward_angle(car_next_state,
                                                                                         self.way_points,
                                                                                         check_point_index)
        self.sensing_info.track_forward_obstacles = self.airsim_env.get_track_forward_obstacle(car_next_state,
                                                                                               self.way_points,
                                                                                               check_point_index,
                                                                                               self.all_obstacles)
        self.sensing_info.distance_to_way_points = self.airsim_env.get_distance_to_way_points(car_next_state, self.way_points,
                                                                              check_point_index)
        return self.sensing_info


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

if __name__ == "__main__":
    tester = DQNRewardTester()
    tester.run()
