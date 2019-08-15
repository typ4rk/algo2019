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
    def get_baseline(self, sensing_info):
        thresh_dist = self.half_road_limit

        # baseline = 0
        baseline_info = []
        DISTANCE_TO_AVOID = 2.3
        DISTANCE_TO_BASELINE = 2.5
        if len(sensing_info.track_forward_obstacles) > 0:
            o_dist, o_to_middle = sensing_info.track_forward_obstacles[0]
            if o_dist < 50:
                # avoid_o_to_middle = abs(sensing_info.to_middle - o_to_middle)
                # center position
                # if abs(o_to_middle) < 1:
                #     avoid_section_left = abs(o_to_middle - 2)
                #     avoid_section_right = abs(o_to_middle + 2)
                #     baseline = max(avoid_section_left, avoid_section_right)
                #     print("[Obstable] baseline: ", baseline, ", o_to_middle: ", o_to_middle) 

                half_dist = (abs(o_to_middle) + thresh_dist) / 3 
                if o_to_middle < -1:
                    baseline = o_to_middle + half_dist
                    # print("move to Right!! baseline: ", baseline, "obs: ", o_to_middle, "car: ", sensing_info.to_middle)
                elif o_to_middle > 1:
                    baseline = o_to_middle - half_dist
                    # print("move to Left!! baseline: ", baseline, "obs: ", o_to_middle, "car: ", sensing_info.to_middle)
                else:
                    if sensing_info.to_middle >= 0:
                        baseline = o_to_middle + half_dist
                        # print("Center-Right!! baseline: ", baseline, "obs: ", o_to_middle, "car: ", sensing_info.to_middle)
                    else:
                        baseline = o_to_middle - half_dist
                        # print("Center-Left!! baseline: ", baseline, "obs: ", o_to_middle, "car: ", sensing_info.to_middle)

                thresh_left = o_to_middle - DISTANCE_TO_AVOID
                thresh_right = o_to_middle + DISTANCE_TO_AVOID                
                baseline_info.append(baseline)
                baseline_info.append(thresh_left)
                baseline_info.append(thresh_right)

        #return baseline    
        return baseline_info

    def compute_reward(self, sensing_info):

        # =========================================================== #
        # Area for writing code
        # =========================================================== #
        
        thresh_dist = self.half_road_limit  # 4 wheels off the track
        dist = abs(sensing_info.to_middle)
        DISTANCE_DECAY_RATE = 1.2        # The rate at which the reward decays for the distance function
        CENTER_SPEED_MULTIPLIER = 2.0    # The ratio at which we prefer the distance reward to the speed reward
        # avoid_o_to_middle = 10
        baseline = 0

        # sensing_info:
        # sensing_info.collided
        # sensing_info.speed
        # sensing_info.moving_forward
        # sensing_info.moving_angle
        # sensing_info.lap_progress
        # sensing_info.track_forward_angles
        # sensing_info.track_forward_obstacles

        # [Obstacles]
        baseline_info = self.get_baseline(sensing_info) 

        # [Speed]
        # compute_speed_reward(sensing_info)

        if dist > thresh_dist:
            reward = -1
        elif sensing_info.collided:
            reward = -1
        elif len(baseline_info) > 0:
            baseline = baseline_info[0] 
            thresh_left = baseline_info[1]
            thresh_right = baseline_info[2]

            # if sensing_info.to_middle > thresh_left and sensing_info.to_middle < thresh_right:
            #     reward = 0
            # else:
            reward = math.exp(-(abs(sensing_info.to_middle - baseline) * DISTANCE_DECAY_RATE))
                # print("baseline:", abs(sensing_info.to_middle - baseline), "calc:", abs(sensing_info.to_middle - baseline) * DISTANCE_DECAY_RATE)
            
            print("[Reward] ", round(reward,3), "dist: ", round(sensing_info.to_middle, 2), ", [base]", round(baseline, 2), "L:", round(thresh_left,2), "R:", round(thresh_right, 2))
        else:
            # if dist > 5:
            #     reward = 0.1
            # elif dist > 4:
            #     reward = 0.2
            # elif dist > 3:
            #     reward = 0.4 + up_speed_reward
            # elif dist > 2:
            #     reward = 0.6 + up_speed_reward
            # elif dist > 1:
            #     reward = 0.8 + down_speed_reward + up_speed_reward
            # else:
            #     reward = 1 + down_speed_reward*2 + up_speed_reward*2
            reward = math.exp(-(dist * DISTANCE_DECAY_RATE))
            print("[Reward] ", round(reward,3), ", dist: ", round(dist, 2))

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
        # 조금 주행?? ?�킨??.
        self.make_initial_movement(self.car_controls, self.client)
        car_current_state = self.client.getCarState(self.player_name)
        backed_car_state = car_current_state

        check_point_index = 0

        # while 루프?�작.
        while True:
            # ?�재 ?�태 구성
            car_current_state = self.client.getCarState(self.player_name)

            check_point_index, _ = self.airsim_env.get_current_way_points(car_current_state, self.way_points,
                                                                          check_point_index)

            # ?�싱 ?�이?? 계산
            sensing_info = self.calc_sensing_data(car_current_state, car_prev_state, backed_car_state,
                                                  self.way_points,
                                                  check_point_index)

            agent_current_state = self.airsim_env.get_current_state(car_current_state, car_prev_state, self.way_points,
                                                                    check_point_index, self.all_obstacles)
            
            # #print("[state]forward_angle: {}".format(agent_current_state[0]))  
            # print("[state]change_rate: index: {} (max: {})".format(agent_current_state[0], agent_current_state[1]))
            # print("[state]moving_angle: {}".format(agent_current_state[2]))
            # print("[state]dist: {}".format(agent_current_state[3]))
            # print("[state]speed: {}".format(agent_current_state[4]))
            # print("[state]o_dist: {}, o_to_middle: {}".format(agent_current_state[5], agent_current_state[6]))            
            # #print(agent_current_state)

            # 보상 ?�수�? ?�라미터�? ?�겨준??.
            reward = self.compute_reward(sensing_info)
            print("[REWARD] value : {}".format(reward))

            if round(self.car_current_pos_x, 4) != round(self.car_next_pos_x, 4):
                backed_car_state = car_current_state
            car_prev_state = car_current_state

            if is_debug:
                print("---------")
                print("to middle: {}".format(sensing_info.to_middle))

                #print("collided: {}".format(sensing_info.collided))
                print("car speed: {} km/h".format(sensing_info.speed))

                #print("is moving forward: {}".format(sensing_info.moving_forward))
                print("moving angle: {}".format(sensing_info.moving_angle))
                print("lap_progress: {}".format(sensing_info.lap_progress))

                print("track_forward_angles: {}".format(sensing_info.track_forward_angles))
                print("track_forward_obstacles: {}".format(sensing_info.track_forward_obstacles))
				#print("distance_to_way_points: {}".format(sensing_info.distance_to_way_points))

                if len(sensing_info.track_forward_obstacles) > 0:
                    o_dist, o_to_middle = sensing_info.track_forward_obstacles[0]
                    if o_dist < 50:
                        avoid_o_to_middle = abs(sensing_info.to_middle - o_to_middle)
                        print("avoid_to_middle: {}". format(avoid_o_to_middle))

                # if len(sensing_info.track_forward_angles) > 0:
                #     diff_angles = abs(sensing_info.track_forward_angles - sensing_info.moving_angle)
                #     print("diff_angles: {}". format(diff_angles))

                up_speed = True
                up_speed_reward = 0
                down_speed_reward = 0

                # ?�방 주행각도 변?�량 ?�보
                change_rate_angles = []
                for x in range(0, 9):
                    change_value = abs(sensing_info.track_forward_angles[x+1] - sensing_info.track_forward_angles[x])
                    change_rate_angles.append(change_value)
                    if x < 3 and change_value > 15:
                        up_speed = False

                    #print("-change_rate: {}".format())
                print("change_rate: {}".format(change_rate_angles))

                max_change_value = max(change_rate_angles)
                max_change_index = change_rate_angles.index(max_change_value)
        
                # 커브각도가 15 이상인 코너링 구간에 근접한 경우
                if max_change_value > 15:
                    if max_change_index < 4 and max_change_index > 0:
                        up_speed = False                        

                if up_speed == True:
                    print("up_speed !!")
                else:
                    print("down_speed !!")

                if up_speed == True and sensing_info.speed > 40:
                    up_speed_reward = 0.2
                    print("[EXTRA REWARD]: +0.2 (all), speed: {}".format(sensing_info.speed))
                elif up_speed == False and sensing_info.speed < 30:
                    down_speed_reward = 0.2
                    print("[EXTRA REWARD]: +0.2 (only dist < 1), speed: {}".format(sensing_info.speed))
                    
                #print("is moving forward: {}".format(sensing_info.moving_forward))
                print("moving angle: {}".format(sensing_info.moving_angle))
                #print("lap_progress: {}".format(sensing_info.lap_progress))
                print("=========================================================")

            time.sleep(1.0)
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
