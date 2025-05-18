from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        # Lưu màu của đội riêng cho phần top và phần bottom
        self.team_colors_top = {}
        self.team_colors_bottom = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)
        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color_top(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # Lấy phần top của ảnh
        top_half_image = image[0:int(image.shape[0] / 2), :]
        # Lấy model clustering cho top half
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        # Lấy nhãn từ 4 góc của top half
        corner_clusters = [clustered_image[0, 0],
                           clustered_image[0, -1],
                           clustered_image[-1, 0],
                           clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def get_player_color_bottom(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # Lấy phần bottom của ảnh (từ giữa đến cuối)
        bottom_half_image = image[int(image.shape[0] / 2):, :]
        kmeans = self.get_clustering_model(bottom_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(bottom_half_image.shape[0], bottom_half_image.shape[1])
        corner_clusters = [clustered_image[0, 0],
                           clustered_image[0, -1],
                           clustered_image[-1, 0],
                           clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        top_colors = []
        bottom_colors = []
        for _, detection in player_detections.items():
            bbox = detection["bbox"]
            top_color = self.get_player_color_top(frame, bbox)
            bottom_color = self.get_player_color_bottom(frame, bbox)
            top_colors.append(top_color)
            bottom_colors.append(bottom_color)
        
        # Phân cụm riêng cho phần top
        kmeans_top = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans_top.fit(top_colors)
        self.team_colors_top[1] = kmeans_top.cluster_centers_[0]
        self.team_colors_top[2] = kmeans_top.cluster_centers_[1]
        self.kmeans_top = kmeans_top

        # Phân cụm riêng cho phần bottom
        kmeans_bottom = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans_bottom.fit(bottom_colors)
        self.team_colors_bottom[1] = kmeans_bottom.cluster_centers_[0]
        self.team_colors_bottom[2] = kmeans_bottom.cluster_centers_[1]
        self.kmeans_bottom = kmeans_bottom
        self.team_colors = self.team_colors_top


    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Tính màu riêng của phần top và bottom
        top_color = self.get_player_color_top(frame, player_bbox)
        bottom_color = self.get_player_color_bottom(frame, player_bbox)
        
        # Dự đoán đội dựa trên từng phần
        team_top = self.kmeans_top.predict(top_color.reshape(1, -1))[0] + 1
        team_bottom = self.kmeans_bottom.predict(bottom_color.reshape(1, -1))[0] + 1

        # Quy tắc quyết định: nếu cả 2 dự đoán giống nhau -> dùng kết quả đó, nếu không ưu tiên phần top
        if team_top == team_bottom:
            team = team_top
        else:
            team = team_top

        # Ví dụ: nếu player_id == 86 thì mặc định cho đội 1
        if player_id == 86:
            team = 1

        self.player_team_dict[player_id] = team
        return team





# from sklearn.cluster import KMeans
# import numpy as np

# def get_foot_position(bbox):
#     x1, y1, x2, y2 = bbox
#     x_center = (x1 + x2) / 2
#     return np.array([x_center, y2], dtype=np.float32)

# class TeamAssigner:
#     def __init__(self, radius_min=20, radius_max=200, radius_step=10):
#         self.radius_min = radius_min
#         self.radius_max = radius_max
#         self.radius_step = radius_step
#         self.player_team_dict = {}  # Lưu team của từng cầu thủ: key là player_id, value là team (1 hoặc 2)
#         self.team_colors_top = {}   # Lưu màu trung tâm (phần top) của team 1 và team 2
#         self.team_colors_bottom = {}# Lưu màu trung tâm (phần bottom) của team 1 và team 2
#         self.n_clusters = 4         # Sử dụng 4 cụm để phân chia: 2 cụm lớn (đội) và 2 cụm nhỏ (thủ môn)
#         self.team1_label = None
#         self.team2_label = None
#         self.kmeans_top = None
#         self.kmeans_bottom = None

#     def get_clustering_model(self, image, n_clusters=2):
#         image_2d = image.reshape(-1, 3)
#         kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1)
#         kmeans.fit(image_2d)
#         return kmeans

#     def get_player_color_top(self, frame, bbox):
#         image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
#         top_half = image[0:int(image.shape[0] / 2), :]
#         kmeans = self.get_clustering_model(top_half, n_clusters=2)
#         labels = kmeans.labels_.reshape(top_half.shape[0], top_half.shape[1])
#         corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
#         non_player_cluster = max(set(corners), key=corners.count)
#         player_cluster = 1 - non_player_cluster
#         return kmeans.cluster_centers_[player_cluster]

#     def get_player_color_bottom(self, frame, bbox):
#         image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
#         bottom_half = image[int(image.shape[0] / 2):, :]
#         kmeans = self.get_clustering_model(bottom_half, n_clusters=2)
#         labels = kmeans.labels_.reshape(bottom_half.shape[0], bottom_half.shape[1])
#         corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
#         non_player_cluster = max(set(corners), key=corners.count)
#         player_cluster = 1 - non_player_cluster
#         return kmeans.cluster_centers_[player_cluster]

#     def assign_team_color(self, frame, player_detections):
#         top_colors = []
#         bottom_colors = []
#         player_positions = []
#         player_ids = []

#         for player_id, detection in player_detections.items():
#             bbox = detection["bbox"]
#             top_color = self.get_player_color_top(frame, bbox)
#             bottom_color = self.get_player_color_bottom(frame, bbox)
#             if top_color is None or bottom_color is None:
#                 continue
#             top_colors.append(top_color)
#             bottom_colors.append(bottom_color)
#             player_positions.append(get_foot_position(bbox))
#             player_ids.append(player_id)

#         if len(top_colors) < self.n_clusters:
#             return

#         top_colors = np.array(top_colors)
#         # Clustering cho phần top với n_clusters=4
#         kmeans_top = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init=10)
#         kmeans_top.fit(top_colors)
#         labels_top = kmeans_top.labels_
#         counts_top = {}
#         for lbl in labels_top:
#             counts_top[lbl] = counts_top.get(lbl, 0) + 1
#         sorted_top = sorted(counts_top.items(), key=lambda x: x[1], reverse=True)
#         self.team1_label = sorted_top[0][0]
#         self.team2_label = sorted_top[1][0]
#         gk_clusters_top = [sorted_top[2][0], sorted_top[3][0]]
#         team_top = {}
#         for i, lbl in enumerate(labels_top):
#             pid = player_ids[i]
#             if lbl == self.team1_label:
#                 team_top[pid] = 1
#             elif lbl == self.team2_label:
#                 team_top[pid] = 2
#             else:
#                 team_top[pid] = None

#         bottom_colors = np.array(bottom_colors)
#         # Clustering cho phần bottom với n_clusters=4
#         kmeans_bottom = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init=10)
#         kmeans_bottom.fit(bottom_colors)
#         labels_bottom = kmeans_bottom.labels_
#         counts_bottom = {}
#         for lbl in labels_bottom:
#             counts_bottom[lbl] = counts_bottom.get(lbl, 0) + 1
#         sorted_bottom = sorted(counts_bottom.items(), key=lambda x: x[1], reverse=True)
#         team_clusters_bottom = [sorted_bottom[0][0], sorted_bottom[1][0]]
#         gk_clusters_bottom = [sorted_bottom[2][0], sorted_bottom[3][0]]
#         team_bottom = {}
#         for i, lbl in enumerate(labels_bottom):
#             pid = player_ids[i]
#             if lbl in team_clusters_bottom:
#                 team_bottom[pid] = 1 if lbl == team_clusters_bottom[0] else 2
#             else:
#                 team_bottom[pid] = None

#         pos_dict = {pid: pos for pid, pos in zip(player_ids, player_positions)}

#         # Với các candidate thủ môn (None), gán lại team dựa trên vòng tròn tăng dần
#         for pid in player_ids:
#             if team_top[pid] is None:
#                 team_top[pid] = self.reassign_goalkeeper(pid, pos_dict, team_top)
#             if team_bottom[pid] is None:
#                 team_bottom[pid] = self.reassign_goalkeeper(pid, pos_dict, team_bottom)

#         self.kmeans_top = kmeans_top
#         self.kmeans_bottom = kmeans_bottom
#         # Kết hợp kết quả của phần top và bottom (nếu trùng, dùng kết quả đó; nếu khác, ưu tiên top)
#         for pid in player_ids:
#             if team_top[pid] == team_bottom[pid]:
#                 team = team_top[pid]
#             else:
#                 team = team_top[pid]
#             # Lưu kết quả phân loại cho từng cầu thủ
#             self.player_team_dict[pid] = team

#         self.team_colors_top = {1: kmeans_top.cluster_centers_[self.team1_label],
#                                 2: kmeans_top.cluster_centers_[self.team2_label]}
#         self.team_colors_bottom = {1: kmeans_bottom.cluster_centers_[team_clusters_bottom[0]],
#                                    2: kmeans_bottom.cluster_centers_[team_clusters_bottom[1]]}

#         self.team_colors = self.team_colors_top
#         return team


#     # def reassign_goalkeeper(self, candidate_pid, pos_dict, assignments):
#     #     radius = self.radius_min
#     #     while radius <= self.radius_max:
#     #         count_team1 = 0
#     #         count_team2 = 0
#     #         for pid, pos in pos_dict.items():
#     #             if pid == candidate_pid:
#     #                 continue
#     #             if np.linalg.norm(pos - pos_dict[candidate_pid]) <= radius:
#     #                 team = assignments.get(pid)
#     #                 if team == 1:
#     #                     count_team1 += 1
#     #                 elif team == 2:
#     #                     count_team2 += 1
#     #         if (count_team1 >= 3 or count_team2 >= 3) and (count_team1 != count_team2):
#     #             team = 1
#     #         else: team = 2

#     #         radius += self.radius_step
#     #     return team

#     def reassign_goalkeeper(self, candidate_pid, pos_dict, assignments):
#         found_team = None  # Chưa tìm được đội
#         radius = self.radius_min
#         while radius <= self.radius_max:
#             count_team1 = 0
#             count_team2 = 0
#             for pid, pos in pos_dict.items():
#                 if pid == candidate_pid:
#                     continue
#                 dist = np.linalg.norm(pos - pos_dict[candidate_pid])
#                 if dist <= radius:
#                     team = assignments.get(pid)
#                     if team == 1:
#                         count_team1 += 1
#                     elif team == 2:
#                         count_team2 += 1

#             # Kiểm tra điều kiện
#             if (count_team1 >= 3) and (count_team1 >count_team2):
#                 found_team = 1
#                 break
#             elif (count_team2 >= 3) and (count_team1 < count_team2):
#                 found_team = 2
#                 break

#             radius += self.radius_step

#         # Nếu không tìm được đội (found_team vẫn None), gán mặc định team 1
#         if found_team is None:
#             found_team = 1

#         return found_team


#     def get_player_team(self, frame, player_bbox, player_id):
#         # Nếu player đã được gán team và không None, trả về nó
#         team = self.player_team_dict.get(player_id, None)
#         if team is None:
#             # Nếu chưa được gán, gán mặc định team 1
#             team = 1
#             self.player_team_dict[player_id] = team
#         return team


