from sklearn.cluster import KMeans
from rubik import RubikBB  # Import RubikBB
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

        self.rubik = RubikBB(n_clusters=2)  # RubikBB
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        
        #Gán màu cho mỗi đội dựa trên 4 vùng màu của áo.
        jersey_colors_list = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            # Thay đổi: Sử dụng RubikBB để phân loại áo thành 4 vùng
            jersey_colors = self.rubik.classify_jersey_colors(frame, bbox)
            # Thay đổi: Thêm 4 mã màu vào danh sách
            jersey_colors_list.append(jersey_colors)  

        # Thay đổi: Tính trung bình 4 mã màu của mỗi cầu thủ
        avg_jersey_colors = np.array([np.mean(colors, axis=0) for colors in jersey_colors_list])

        # Thay đổi: Phân cụm KMeans trên các mã màu trung bình
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(avg_jersey_colors)
        self.kmeans = kmeans  # Lưu lại model KMeans để sử dụng trong get_player_team

        # Thay đổi: Gán 2 màu trung bình làm màu đội
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
         # Thay đổi: Sử dụng RubikBB để phân loại áo thành 4 vùng
        jersey_colors = self.rubik.classify_jersey_colors(frame, player_bbox)  # Thêm mới
 
        # Lấy nhãn của cụm phổ biến nhất trong 4 vùng
        labels = []
        for color in jersey_colors:
            label = self.kmeans.predict(color.reshape(1, -1))[0]
            labels.append(label)
        dominant_label = max(set(labels), key=labels.count)

        # Lấy màu trung bình của cụm chủ đạo
        dominant_color = self.kmeans.cluster_centers_[dominant_label]

        # So sánh dominant_color với màu sắc của các đội
        distances = [np.linalg.norm(dominant_color - np.array(team_color)) 
                    for team_color in self.team_colors.values()]
        team_id = np.argmin(distances) + 1  # +1 vì team_id bắt đầu từ 1

        if player_id ==86:
            team_id=2

        self.player_team_dict[player_id] = team_id

        return team_id
