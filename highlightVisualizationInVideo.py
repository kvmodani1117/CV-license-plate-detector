import ast

import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # List of tuples containing the points and the direction of the line length for each line
    points_and_directions = [
        ((x1, y1), (0, line_length_y)),       # Top-left vertical
        ((x1, y1), (line_length_x, 0)),       # Top-left horizontal
        ((x1, y2), (0, -line_length_y)),      # Bottom-left vertical
        ((x1, y2), (line_length_x, 0)),       # Bottom-left horizontal
        ((x2, y1), (-line_length_x, 0)),      # Top-right horizontal
        ((x2, y1), (0, line_length_y)),       # Top-right vertical
        ((x2, y2), (0, -line_length_y)),      # Bottom-right vertical
        ((x2, y2), (-line_length_x, 0))       # Bottom-right horizontal
    ]

    # Draw each line based on the starting point and direction
    for (start_x, start_y), (delta_x, delta_y) in points_and_directions:
        end_x, end_y = start_x + delta_x, start_y + delta_y
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

    return img



results = pd.read_csv('./data_records.csv')
# results = pd.read_csv('./test_interpolated.csv')

# print("results------------->", results)

# load video
video_path = 'carVideo.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
outputVideo = cv2.VideoWriter('./outputVideo.mp4', fourcc, fps, (width, height))



license_plate = {}
for carId in np.unique(results['carId']):
    car_results = results[results['carId'] == carId]
    max_confidence = car_results['licenseTextConfidenceScore'].max()
    max_confidence_row = car_results[car_results['licenseTextConfidenceScore'] == max_confidence].iloc[0]

    license_plate[carId] = {
        'license_crop': None,
        'license_plate_number': max_confidence_row['licenseText']
    }

    cap.set(cv2.CAP_PROP_POS_FRAMES, max_confidence_row['frameNo'])
    ret, frame = cap.read()

    # Cleaning up the bounding box string and converting it to a tuple
    bbox_str = max_confidence_row['licenseBoundingBox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
    x1, y1, x2, y2 = ast.literal_eval(bbox_str)

    # Ensure coordinates are integer
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # Cropping and resizing the license plate image
    license_crop = frame[y1:y2, x1:x2]
    resized_width = int((x2 - x1) * 400 / (y2 - y1))
    license_crop = cv2.resize(license_crop, (resized_width, 400))

    license_plate[carId]['license_crop'] = license_crop



frameNo = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frameNo += 1
    if ret:
        df_ = results[results['frameNo'] == frameNo]
        for row_indx in range(len(df_)):
            # draw car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['carBoundingBox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['licenseBoundingBox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # crop license plate
            license_crop = license_plate[df_.iloc[row_indx]['carId']]['license_crop']

            H, W, _ = license_crop.shape

            try:
                
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                        int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop
                

                license_plate_text = license_plate[df_.iloc[row_indx]['carId']]['license_plate_number']
                font_scale = 4.5
                thickness = 20
                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]['carId']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    thickness)

                cv2.putText(frame,
                            license_plate[df_.iloc[row_indx]['carId']]['license_plate_number'],
                            (text_width, text_height),
                            # (int((car_x2 + car_x1 - text_width) / 4), int(car_y1 - H - 250 + (text_height / 4))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (0, 0, 0),
                            thickness)

            except:
                pass

        outputVideo.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)


outputVideo.release()
cap.release()