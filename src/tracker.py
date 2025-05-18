from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from .bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    @staticmethod
    def calculate_transition_counts_and_correct(ball_control_ids, team_assignments):
        total_transitions = {1: 0, 2: 0}
        correct_transitions = {1: 0, 2: 0}

        if len(ball_control_ids) < 2:
            return total_transitions, correct_transitions

        for i in range(1, len(ball_control_ids)):
            prev_id = ball_control_ids[i - 1]
            curr_id = ball_control_ids[i]
            prev_team = team_assignments.get(prev_id, None)
            curr_team = team_assignments.get(curr_id, None)
            
            # Nếu có chuyển giao bóng (id thay đổi) và team của frame trước có giá trị
            if prev_id != curr_id and prev_team is not None:
                total_transitions[prev_team] += 1
                # Nếu team của frame i cũng giống team của frame i-1, chuyển giao chính xác
                if curr_team == prev_team:
                    correct_transitions[prev_team] += 1

        return total_transitions, correct_transitions
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40 
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Vẽ hình chữ nhật mờ
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Xác định đội đang kiểm soát bóng
        team_with_ball = team_ball_control[frame_num] if frame_num < len(team_ball_control) else None

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]

        # Tính toán số lượng frame mỗi đội kiểm soát bóng
        team_1_num_frames = np.count_nonzero(team_ball_control_till_frame == 1)
        team_2_num_frames = np.count_nonzero(team_ball_control_till_frame == 2)

        # Tránh lỗi chia cho 0
        if team_1_num_frames + team_2_num_frames == 0:
            team_1_percentage = 0.0
            team_2_percentage = 0.0
        else:
            team_1_percentage = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
            team_2_percentage = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        # Thay đổi màu chữ trong bảng thống kê
        team_1_color = (0, 255, 0) if team_with_ball == 1 else (0, 0, 0)
        team_2_color = (0, 255, 0) if team_with_ball == 2 else (0, 0, 0)

        # Hiển thị thông tin kiểm soát bóng
        cv2.putText(frame, f"Team 1 Ball Control: {team_1_percentage * 100:.2f}%", (1400, 900),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, team_1_color, 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_percentage * 100:.2f}%", (1400, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, team_2_color, 3)

        return frame
    
    def draw_team_transition_accuracy(self, frame, frame_num, ball_control_ids, team_assignments):
        # Lấy danh sách id cầu thủ kiểm soát bóng từ đầu đến frame hiện tại
        partial_ids = ball_control_ids[:frame_num + 1]
        # Tính total_transitions và correct_transitions cho partial_ids
        total_transitions, correct_transitions = self.calculate_transition_counts_and_correct(partial_ids, team_assignments)

        # Vẽ nền hình chữ nhật mờ cho bảng thống kê
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 850), (600, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Tính tỷ lệ chuyển giao chính xác cho mỗi đội
        ratio_team1 = correct_transitions[1] / total_transitions[1] if total_transitions[1] > 0 else 0.0
        ratio_team2 = correct_transitions[2] / total_transitions[2] if total_transitions[2] > 0 else 0.0

        # Chọn màu hiển thị (màu xanh dương)
        team_1_color = (255, 0, 0)
        team_2_color = (255, 0, 0)

        # Hiển thị thông tin Transition Accuracy lên frame
        cv2.putText(frame, f"Team 1 Transition Accuracy: {ratio_team1 * 100:.2f}%", 
                    (10, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, team_1_color, 3)
        cv2.putText(frame, f"Team 2 Transition Accuracy: {ratio_team2 * 100:.2f}%", 
                    (10, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, team_2_color, 3)

        return frame
        # # Vẽ nền hình chữ nhật mờ cho bảng thống kê
        # overlay = frame.copy()
        # cv2.rectangle(overlay, (0, 850), (600, 970), (255, 255, 255), -1)
        # alpha = 0.4
        # cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # # Tính tỷ lệ chuyển giao chính xác cho mỗi đội:
        # ratio_team1 = correct_transitions[1] / total_transitions[1] if total_transitions[1] > 0 else 0.0
        # ratio_team2 = correct_transitions[2] / total_transitions[2] if total_transitions[2] > 0 else 0.0


        # # Sử dụng màu xanh dương để hiển thị
        # team_1_color = (255, 0, 0)
        # team_2_color = (255, 0, 0)

        # # Hiển thị thông tin Transition Accuracy lên frame
        # cv2.putText(frame, f"Team 1 Transition Accuracy: {ratio_team1 * 100:.2f}%", 
        #             (10, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, team_1_color, 3)
        # cv2.putText(frame, f"Team 2 Transition Accuracy: {ratio_team2 * 100:.2f}%", 
        #             (10, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, team_2_color, 3)

        # return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control, ball_control_ids, team_assignments):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            #Draw Team Transition Accuracy
            frame = self.draw_team_transition_accuracy(frame, frame_num, ball_control_ids, team_assignments)

            output_video_frames.append(frame)

        return output_video_frames
    
    @staticmethod
    def write_final_info(info_path, team_ball_control, ball_control_ids, team_assignments):
         # Tính tỷ lệ kiểm soát bóng dựa trên toàn bộ frame có giá trị 1 hoặc 2
        team_ball_control = np.array(team_ball_control)
        team_1_num_frames = np.count_nonzero(team_ball_control == 1)
        team_2_num_frames = np.count_nonzero(team_ball_control == 2)

        if team_1_num_frames + team_2_num_frames == 0:
            team_1_percentage = 0.0
            team_2_percentage = 0.0
        else:
            team_1_percentage = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
            team_2_percentage = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        # Tính số chuyển giao và chuyển giao chính xác
        total_transitions, correct_transitions = Tracker.calculate_transition_counts_and_correct(ball_control_ids, team_assignments)

        # Tính tỉ lệ đường chuyền chính xác (pass accuracy)
        accuracy_team1 = (correct_transitions[1] / total_transitions[1] * 100) if total_transitions[1] > 0 else 0
        accuracy_team2 = (correct_transitions[2] / total_transitions[2] * 100) if total_transitions[2] > 0 else 0

        # Ghi kết quả ra file info
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("Final Ball Control Ratios:\n")
            f.write("Team 1: {:.2f}%\n".format(team_1_percentage * 100))
            f.write("Team 2: {:.2f}%\n\n".format(team_2_percentage * 100))
            f.write("Transition Counts:\n")
            f.write("Team 1 - Total Transitions: {}, Correct Transitions: {}\n".format(total_transitions[1], correct_transitions[1]))
            f.write("Team 2 - Total Transitions: {}, Correct Transitions: {}\n\n".format(total_transitions[2], correct_transitions[2]))
            f.write("Transition Accuracy:\n")
            f.write("Team 1 Transition Accuracy: {:.2f}%\n".format(accuracy_team1))
            f.write("Team 2 Transition Accuracy: {:.2f}%\n".format(accuracy_team2))