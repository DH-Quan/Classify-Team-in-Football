from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    # Đọc video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Khởi tạo Tracker
    tracker = Tracker('models/best.pt')

    # Theo dõi đối tượng trong video
    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=True,
                                        stub_path='stubs/track_stubs.pkl')
    
    # Thêm thông tin vị trí vào tracks
    tracker.add_position_to_tracks(tracks)

    # Khởi tạo CameraMovementEstimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])

    # Ước tính chuyển động của camera
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    
    # Điều chỉnh vị trí của các đối tượng dựa trên chuyển động của camera
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Khởi tạo ViewTransformer
    view_transformer = ViewTransformer()

    # Biến đổi phối cảnh của các vị trí
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Nội suy vị trí của bóng
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Khởi tạo SpeedAndDistance_Estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()

    # Tính toán tốc độ và khoảng cách
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Khởi tạo TeamAssigner
    team_assigner = TeamAssigner()

    # Phân loại đội bóng ban đầu
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    # Mở file để ghi log
    log_file = 'output_videos/player_teams.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        # Lặp qua từng frame
        for frame_num, player_track in enumerate(tracks['players']):
            if frame_num % 24 == 0:
                team_assigner.sort_players_to_teams(video_frames[frame_num], tracks['players'][frame_num])

            # Lặp qua từng cầu thủ trong frame
            for player_id, track in player_track.items():
                # Xác định đội của cầu thủ
                team = team_assigner.get_player_team(video_frames[frame_num],
                                                            track['bbox'],
                                                            player_id)
                # Lưu thông tin đội bóng của cầu thủ
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
                # Ghi thông tin vào file log
                f.write(f"Frame {frame_num} - Player {player_id}: team {team}\n")

        # Assign Ball Aquisition
        player_assigner = PlayerBallAssigner()
        team_ball_control = []

        # Lặp qua từng frame
        for frame_num, player_track in enumerate(tracks['players']):
            # Kiểm tra xem có bóng trong frame không
            if 1 in tracks['ball'][frame_num]:
                ball_bbox = tracks['ball'][frame_num][1]['bbox']
                # Xác định cầu thủ đang kiểm soát bóng
                assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
                if assigned_player != -1:
                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
                else:
                    # Nếu không có cầu thủ nào kiểm soát bóng, giữ nguyên đội kiểm soát bóng trước đó
                    if team_ball_control:
                        team_ball_control.append(team_ball_control[-1])
                    else:
                        team_ball_control.append(None)
            else:
                # Xử lý trường hợp không tìm thấy bóng
                assigned_player = -1
                if team_ball_control:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(None)

        team_ball_control = np.array(team_ball_control)

        # Vẽ kết quả
        ## Vẽ theo dõi đối tượng
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

        ## Vẽ chuyển động của camera
        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

        ## Vẽ tốc độ và khoảng cách
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Lưu video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()