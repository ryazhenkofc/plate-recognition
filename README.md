# License Plate Recognition System

This project is a License Plate Recognition system that utilizes a combination of computer vision and machine learning techniques to identify and process license plates from video footage.

## Project Structure

The project is organized as follows:

- `main.py`: The main script that provides a GUI for uploading videos, processing them, and exporting the results.
- `models/`: Contains the pre-trained models for license plate detection and object detection.
    - `license_plate_detector.pt`: The model for detecting license plates.
    - `yolov8n.pt`: The YOLO model for object detection.
- `sort/`: Implements the SORT algorithm for object tracking.
    - `sort.py`: The main SORT algorithm implementation.
    - `data/`: Training data for the SORT algorithm.
- `src/`: Contains the core functionality for processing videos and license plates.
    - `get_plates.py`: Functions for extracting license plate data.
    - `interpolate.py`: Interpolation utilities for smoothing bounding box coordinates.
    - `process.py`: Core processing functions for video and license plate recognition.
    - `util.py`: Utility functions.
    - `visualize.py`: Functions for visualizing the results.

## Setup

To set up the project, follow these steps:

1. Ensure Python 3.8 or higher is installed.
2. Install the required Python packages by running `pip install -r requirements.txt`.
3. Download the pre-trained models and place them in the `models/` directory.

## Usage

To use the system, run `main.py` and follow the GUI prompts:

1. Upload a video file.
2. Click "Process Video" to start the license plate recognition process.
3. Export the results to an Excel file or visualize the processed video.

## Dependencies

- filterpy
- scikit-image
- pandas
- ultralytics
- easyocr
- scipy
- lap
- opencv-python

## License

This project is licensed under the MIT License - see the LICENSE file for details.