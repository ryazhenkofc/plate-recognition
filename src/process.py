import cv2
import numpy as np
import csv
import os
import string
import easyocr
from scipy.interpolate import interp1d
from ultralytics import YOLO
from sort.sort import Sort
import sys
import signal
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

import signal
import sys

# Define a signal handler to clean up resources
def signal_handler(sig, frame):
    print("Termination signal received. Cleaning up...")
    # Clean up resources here, if needed
    sys.exit(0)

# Register signal handlers for termination signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()

def license_complies_format(text):
    if len(text) != 7:
        return False

    if all(c in string.ascii_uppercase or c in dict_int_to_char.keys() for c in text):
        return True
    else:
        return False

def format_license(text):
    return ''.join(dict_int_to_char.get(c, c) for c in text)

def read_license_plate(license_plate_crop, reader):
    detections = reader.readtext(license_plate_crop)

    for bbox, text, score in detections:
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score

    return None, None

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    for j, (xcar1, ycar1, xcar2, ycar2, car_id) in enumerate(vehicle_track_ids):
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle_track_ids[j]

    return -1, -1, -1, -1, -1

def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split(" "))) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]

        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row.get('license_plate_bbox_score', '0')
                row['license_number'] = original_row.get('license_number', '0')
                row['license_number_score'] = original_row.get('license_number_score', '0')

            interpolated_data.append(row)

    return interpolated_data

def load_models(coco_model_path, license_plate_model_path):
    try:
        coco_model = YOLO(coco_model_path)
        license_plate_detector = YOLO(license_plate_model_path)
        return coco_model, license_plate_detector
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

def process_frame(frame, coco_model, license_plate_detector, mot_tracker, vehicle_classes, reader):
    frame_results = {}
    detections = coco_model(frame)[0]
    vehicle_detections = [d for d in detections.boxes.data.tolist() if int(d[5]) in vehicle_classes]
    track_ids = mot_tracker.update(np.asarray(vehicle_detections))
    license_plates = license_plate_detector(frame)[0]

    for license_plate in license_plates.boxes.data.tolist():
        car_bbox = get_car(license_plate, track_ids)
        if car_bbox[-1] != -1:
            license_plate_crop = frame[int(license_plate[1]):int(license_plate[3]), int(license_plate[0]):int(license_plate[2])]
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop, reader)
            if license_plate_text:
                frame_results[car_bbox[-1]] = format_result(car_bbox, license_plate, license_plate_text, license_plate_text_score)
    
    return frame_results

def process_license_plate(frame, license_plate):
    x1, y1, x2, y2, score, class_id = license_plate
    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
    return license_plate_crop, license_plate_text, license_plate_text_score

def format_result(car_bbox, license_plate, license_plate_text, license_plate_text_score):
    xcar1, ycar1, xcar2, ycar2, car_id = car_bbox
    x1, y1, x2, y2, score, class_id = license_plate
    return {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
            'license_plate': {'bbox': [x1, y1, x2, y2],
                              'text': license_plate_text,
                              'bbox_score': score,
                              'text_score': license_plate_text_score}}

def process_video(video_path: str) -> int:
    try:
        # Configuration
        reader = easyocr.Reader(['en'])
        vehicle_classes = [2, 3, 5, 7]
        coco_model_path = './models/yolov8n.pt'
        license_plate_model_path = './models/license_plate_detector.pt'
        output_csv_path = './temp.csv'
        vehicle_classes = [2, 3, 5, 7]

        # Error handling - check video path
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return -1
        
        # Load models
        coco_model, license_plate_detector = load_models(coco_model_path, license_plate_model_path)
        if coco_model is None or license_plate_detector is None:
            return -1

        # Object tracker
        mot_tracker = Sort()

        # Video capture
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        results = {}

        # Process video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results[frame_count] = process_frame(frame, coco_model, license_plate_detector, mot_tracker, vehicle_classes, reader)

        cap.release()

        # Write temporary CSV with raw data
        write_csv(results, output_csv_path)

        # Read and interpolate data
        with open(output_csv_path, 'r') as file:
            reader = csv.DictReader(file)
            data = list(reader)
            interpolated_data = interpolate_bounding_boxes(data)

        # Write final CSV with interpolated data
        header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
        with open('final.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(interpolated_data)

        # Remove temporary file
        os.remove(output_csv_path)

        return 0

    except Exception as e:
        print(f"Error processing video: {e}")
        return -1
