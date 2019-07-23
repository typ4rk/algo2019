#forward_angle_arr = [-2, -3, -3, -3, -2, -1, 1, 4, 8, 13]
# [-1, 0, 0, 1, 1, 2, 3, 4, 5]
# index: 8 (max: 5)
# forward_angle_arr = [17, 25, 29, 31, 33, 34, 35, 34, 35, 36]
# [8, 4, 2, 2, 1, 1, -1, 1, 1]
# index: 0 (max: 8)
forward_angle_arr = [-17, -25, -29, -31, -33, -34, -35, -34, -35, -36]
# [8, 4, 2, 2, 1, 1, 1, 1, 1]
# index: 0 (max: -8)
change_rate = []
for x in range(0, 9):
    change_rate.append(abs(forward_angle_arr[x+1] - forward_angle_arr[x]))
 
print(change_rate)
max_change_value = max(change_rate)
max_change_index = change_rate.index(max_change_value)
# print("index: {} (max: {})".format(max_change_index, max_change_value))
# print(change_rate[max_change_index + 1])
if change_rate[max_change_index + 1] - change_rate[max_change_index] < 0:
    max_change_value = max_change_value * -1
print("index: {} (max: {})".format(max_change_index, max_change_value))