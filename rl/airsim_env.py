import setup_path
import math
import numpy as np
from numpy import linalg as LA
from airsim_base_env import AirSimBaseEnv

way_point_unit = 10  # meter


class AirSimEnv(AirSimBaseEnv):

    @staticmethod
    def get_state_size():
        return 7

    def get_current_state(self, car_state, car_prev_state, way_points, check_point, all_obstacles):

        state = []

        # ======
        # (1) Forward angle
        # ======
        # 현재 주행 구간에서 10 개 각도
        forward_angle_arr = self.get_track_forward_angle(car_state, way_points, check_point)

        # forward_angle = round(abs(forward_angle_arr[2]), 2)

        # if forward_angle_arr[2] < 0:
        #     forward_angle = forward_angle * -1

        # state.append(forward_angle)
        change_rate = []
        for x in range(0, 9):
            change_rate.append(abs(forward_angle_arr[x+1] - forward_angle_arr[x]))
        
        
        max_change_value = max(change_rate)
        max_change_index = change_rate.index(max_change_value)

        #print(change_rate)
        #print(max_change_index)
        # print("index: {} (max: {})".format(max_change_index, max_change_value))

        if forward_angle_arr[max_change_index + 1] - forward_angle_arr[max_change_index] < 0:
            max_change_value = max_change_value * -1
        #print("index: {} (max: {})".format(max_change_index, max_change_value))
        state.append(max_change_index)
        state.append(max_change_value)

        # ======
        # (2) Moving angle
        # ======
        angle = self.get_moving_angle(car_prev_state, car_state, way_points, check_point)

        state.append(angle)

        # ======
        # (3) Current distance from center(position)
        # ======
        dist = round(self.get_distance_from_center(car_state, way_points, check_point), 2)

        # road width = 10 m

        if self.is_right_of_center(car_state, way_points, check_point):
            state.append(dist)
        else:
            state.append(dist * -1)

        # ======
        # (4) Current velocity
        # ======
        velocity = self.get_speed(car_state)
        state.append(velocity)

        # ======
        # (5, 6) Obstacle distance, to middle
        # ======
        track_forward_obstacles = self.get_track_forward_obstacle(car_state, way_points, check_point, all_obstacles)
        if len(track_forward_obstacles) > 0:
            o_dist, o_to_middle = track_forward_obstacles[0]
            if o_dist < 50:
                state.append(round(o_dist, 1))
                state.append(round(o_to_middle, 2))
            else:
                state.append(0)
                state.append(0)
        else:
            state.append(0)
            state.append(0)

        return state
