import cv2
from sklearn.cluster import KMeans
import numpy as np

class RubikBB():
    def __init__(self, n_clusters=2, color_space="rgb"):
        self.n_clusters = n_clusters
        self.color_space = color_space.lower()


    def _get_section_color(self, image, x1, y1, x2, y2):
        """
        Hàm phụ trợ để tính toán màu sắc của một vùng ảnh.
        """
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Cắt ra vùng ảnh
        section = image[y1:y2, x1:x2]

        # Reshape the section into 2D array
        section_2d = section.reshape(-1, 3)

        # Perform k-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(section_2d)

        # Get the cluster labels
        labels = kmeans.labels_

        # Reshape the labels into the original section shape
        clustered_section = labels.reshape(section.shape[0], section.shape[1])

        # Get the player cluster (giả định rằng cluster của nền là phổ biến nhất ở 4 góc)
        corner_clusters = [clustered_section[0, 0], clustered_section[0, -1],
                          clustered_section[-1, 0], clustered_section[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Lấy màu trung bình của cluster cầu thủ
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color


    def get_player_color_vector(self, image, bbox):
        xmin, ymin, xmax, ymax = bbox
        mid_x = int((xmin + xmax) / 2)
        mid_y = int((ymin + ymax) / 2)

        color_vector = []
        for x1, y1, x2, y2 in [(xmin, ymin, mid_x, mid_y),
                               (mid_x, ymin, xmax, mid_y),
                               (xmin, mid_y, mid_x, ymax),
                               (mid_x, mid_y, xmax, ymax)]:
            
            color = self._get_section_color(image, x1, y1, x2, y2)  
            color_vector.extend(color)

        return np.array(color_vector)
    

    def classify_players_by_color(self, color_vectors):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        kmeans.fit(color_vectors)
        labels = kmeans.labels_
        return labels
    

    def classify_jersey_colors(self, image, bbox):
        # Phân áo thành 4 vùng dựa trên bounding box.
        xmin, ymin, xmax, ymax = bbox
        mid_x = int((xmin + xmax) / 2)
        mid_y = int((ymin + ymax) / 2)

        jersey_colors = []
        for x1, y1, x2, y2 in [(xmin, ymin, mid_x, mid_y),
                               (mid_x, ymin, xmax, mid_y),
                               (xmin, mid_y, mid_x, ymax),
                               (mid_x, mid_y, xmax, ymax)]:
            color = self._get_section_color(image, x1, y1, x2, y2)  
            jersey_colors.append(color)  

        return np.array(jersey_colors)
    

    def preprocess_image(self, image):
        if self.color_space == "hsv":
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == "lab":
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:  # "rgb" hoặc không hợp lệ
            return image