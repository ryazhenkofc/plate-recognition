from moviepy.editor import VideoFileClip
import pandas as pd

def get_plate_data(video_path):
    # Load the video to get its duration
    video = VideoFileClip(rf"{video_path}")

    # Assuming 'results' is your DataFrame loaded from the CSV
    results = pd.read_csv('./final.csv')

    # Initialize the dictionary
    timestamps = {}

    # Calculate timestamps
    fps = 30  # Assuming 30 FPS, adjust according to your video
    for index, row in results.iterrows():
        frame_nmr = row['frame_nmr']
        car_id = row['car_id']
        license_number = row['license_number']
        seconds = frame_nmr / fps
        timestamp = f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"  # Convert to mm:ss format
        if car_id not in timestamps or license_number != timestamps[car_id]['license_plate_number']:
            timestamps[car_id] = {'appearance': timestamp, 'disappearance': timestamp, 'license_plate_number': license_number}
        else:
            timestamps[car_id]['disappearance'] = timestamp

    # Prepare records for DataFrame
    records = []
    for car_id, times in timestamps.items():
        if times['license_plate_number'] != '0':  # Exclude records without a license plate number
            records.append([times['license_plate_number'], times['appearance'], times['disappearance']])

    # Create DataFrame
    columns = ['Plate Number', 'Appearance Timestamp', 'Disappearance Timestamp']
    df_records = pd.DataFrame(records, columns=columns)

    return df_records