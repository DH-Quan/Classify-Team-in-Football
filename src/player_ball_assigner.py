import sys 
sys.path.append('../')
from config import config
from .bbox_utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        # self.max_player_ball_distance = 70
        self.max_player_ball_distance = config['player_ball_params']['max_player_ball_distance']
    
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player

# import sys 
# sys.path.append('../')
# from config import config
# from .bbox_utils import get_center_of_bbox, measure_distance

# class PlayerBallAssigner():
#     def __init__(self):
#         self.max_player_ball_distance = config['player_ball_params']['max_player_ball_distance']
#         self.last_confirmed_player = -1  # Cầu thủ sở hữu bóng cuối cùng
#         self.possession_frames_threshold = 5  # số frame tối thiểu để xác nhận sở hữu
#         self.player_possession_counter = {}  # lưu số frame sở hữu liên tiếp

#     def assign_ball_to_player(self, players, ball_bbox):
#         ball_position = get_center_of_bbox(ball_bbox)

#         assigned_player = -1
#         min_distance = float('inf')

#         # Tìm cầu thủ gần nhất nằm trong ngưỡng khoảng cách
#         for player_id, player in players.items():
#             player_bbox = player['bbox']
#             distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
#             distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
#             distance = min(distance_left, distance_right)

#             if distance < self.max_player_ball_distance and distance < min_distance:
#                 min_distance = distance
#                 assigned_player = player_id

#         # Cập nhật trạng thái giữ bóng
#         if assigned_player != -1:
#             self.player_possession_counter[assigned_player] = self.player_possession_counter.get(assigned_player, 0) + 1

#             # Reset số lần giữ bóng liên tiếp cho các cầu thủ khác
#             for other_player in players.keys():
#                 if other_player != assigned_player:
#                     self.player_possession_counter[other_player] = 0

#             # Xác nhận quyền kiểm soát nếu đủ frame
#             if self.player_possession_counter[assigned_player] >= self.possession_frames_threshold:
#                 self.last_confirmed_player = assigned_player
#         else:
#             # Không ai đủ gần, reset counter
#             for player in players.keys():
#                 self.player_possession_counter[player] = 0

#         return self.last_confirmed_player
