from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner    
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
# from rubik import RubikBB

def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #Initialize Track
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    
    #save cropped image of a player
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]
    #     #crop bbox from frame 
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    #     #save the cropped image 
    #     cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)
    #     break

    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

     # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # # Khởi tạo đối tượng Rubik
    # rubik = Rubik(n_clusters=2, color_space="rgb")

    # for frame_num, frame in enumerate(video_frames):

    #     # Tiền xử lý hình ảnh
    #     processed_image = rubik.preprocess_image(frame)  

    #     # Trích xuất vector đặc trưng màu sắc và phân loại cầu thủ
    #     color_vectors = []  
    #     for player_id, player in tracks['players'][frame_num].items():  
    #         bbox = player['bbox']  
    #         color_vector = rubik.get_player_color_vector(processed_image, bbox)  
    #         color_vectors.append(color_vector)  
    #     team_labels = rubik.classify_players_by_color(color_vectors)  

    #     # Gán nhãn đội cho cầu thủ trong biến tracks
    #     for i, player_id in enumerate(tracks['players'][frame_num].keys()):  
    #         tracks['players'][frame_num][player_id]['team'] = team_labels[i]  

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    log_file = 'output_videos/player_teams.txt'  # Đường dẫn file log
    with open(log_file, 'w') as f:
    
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
                tracks['players'][frame_num][player_id]['team'] = team 
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

                # Ghi thông tin cầu thủ và đội vào file
                f.write(f"player {player_id}: team {team}\n") 

    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np .array(team_ball_control)

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()