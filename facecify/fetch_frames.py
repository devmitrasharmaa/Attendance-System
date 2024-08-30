import cv2
import os

def extract_frames_from_videos(input_dir, output_dir, frame_count=10):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over all the files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mov'):
            video_path = os.path.join(input_dir, filename)
            video_name = os.path.splitext(filename)[0]
            video_output_dir = os.path.join(output_dir, video_name)
            
            # Create a directory for each video to store the frames
            if not os.path.exists(video_output_dir):
                os.makedirs(video_output_dir)
            
            # Capture the video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / cap.get(cv2.CAP_PROP_FPS)
            print(f'Video: {filename}, Duration: {duration}s, Total Frames: {total_frames}')
            
            # Calculate the interval to capture frames
            interval = max(1, int(total_frames / frame_count))
            
            frame_number = 0
            extracted_frame_count = 0
            while cap.isOpened() and extracted_frame_count < frame_count:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_number % interval == 0:
                    frame_filename = os.path.join(video_output_dir, f'frame_{frame_number}.jpg')
                    cv2.imwrite(frame_filename, frame)
                    extracted_frame_count += 1
                frame_number += 1
            
            cap.release()
            print(f'Extracted {extracted_frame_count} frames from {filename}')
    
    print('Frame extraction completed.')

# # Directory containing the dataset videos
# input_videos_directory = r'C:\Users\devmi\OneDrive\Documents\brocode\Python\Flask_shit\Facecify\website1\facecify\static\videos'
# # Directory to store the extracted frames
# output_frames_directory = r'C:\Users\devmi\OneDrive\Documents\brocode\Python\Flask_shit\Facecify\website1\facecify\static\dataset'

# # Extract frames
# extract_frames_from_videos(input_videos_directory, output_frames_directory)
