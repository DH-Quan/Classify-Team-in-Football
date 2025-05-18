from config import config
import cv2
import numpy as np
from src import CameraMovementEstimator, PlayerBallAssigner, SpeedAndDistance_Estimator, TeamAssigner, Tracker, read_video, save_video, ViewTransformer
# from rubik import RubikBB

def main():
    video_frames = read_video(config['paths']['video_input'])

    tracker = Tracker(config['paths']['model_path']) 
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=config['paths']['stubs']['track'])

    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=config['paths']['stubs']['camera_movement'])

    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    log_file = config['paths']['log_team']
    with open(log_file, 'w') as f:
        fps = config['main']['fps']  
        classification_interval = config['main']['classification_interval']

        # for frame_num, player_track in enumerate(tracks['players']):
        #     if frame_num % classification_interval == 0:
        #         for player_id, track in player_track.items():
        #             team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
        #             tracks['players'][frame_num][player_id]['team'] = team
        #             tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
        #             f.write(f"Frame {frame_num} - Player {player_id}: team {team}\n")
        #     else:
        #         for player_id, track in player_track.items():
        #             team = team_assigner.player_team_dict.get(player_id)
        #             if team is not None:
        #                 tracks['players'][frame_num][player_id]['team'] = team
        #                 tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
        #             else:
        #                 tracks['players'][frame_num][player_id]['team'] = None

        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                # Luôn gán team (phân loại lại nếu cần, hoặc lấy từ cache)
                if frame_num % classification_interval == 0:
                    team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                    tracks['players'][frame_num][player_id]['team'] = team
                    tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
                else:
                    team = team_assigner.player_team_dict.get(player_id, None)
                    tracks['players'][frame_num][player_id]['team'] = team
                    if team is not None:
                        tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

                # Luôn ghi ra file (dù ở frame nào)
                f.write(f"Frame {frame_num} - Player {player_id}: team {team}\n")


    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    ball_control_ids = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team = tracks['players'][frame_num][assigned_player].get('team')
            if team is not None:
                team_ball_control.append(team)
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else None)
            ball_control_ids.append(assigned_player)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else None)
            ball_control_ids.append(ball_control_ids[-1] if ball_control_ids else -1)

    team_assignments = team_assigner.player_team_dict
    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control, ball_control_ids, team_assignments)

    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    save_video(output_video_frames, config['paths']['video_output'])

    log_info = config['paths']['log_info']
    Tracker.write_final_info(log_info, team_ball_control, ball_control_ids, team_assignments)


if __name__ == '__main__':
    main()