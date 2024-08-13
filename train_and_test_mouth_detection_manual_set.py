from ultralytics import YOLO
import cv2
import os
import yaml


def edit_yaml_paths(yaml_file_path):
    # Dataset main directory
    root_dir = os.path.dirname(yaml_file_path)

    # Load the YAML file
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Modify YAML file to include correct subset paths
    # Manually labeled dataset only has one subset, so it is used for train test and val
    data['train'] = os.path.join(root_dir, 'manual_dataset/Human_Photos_Labeled')
    data['val'] = os.path.join(root_dir, 'manual_dataset/Human_Photos_Labeled')
    data['test'] = os.path.join(root_dir, 'manual_dataset/Human_Photos_Labeled')

    with open(yaml_file_path, 'w') as file:
        yaml.safe_dump(data, file)

    print("YAML file updated successfully!")


def webcam_predict(input_model, save_boxes=False, conf=0.25):
    cap = cv2.VideoCapture(0)  # 0 as default for webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Width to 1920 for higher resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height to 1080 for higher resolution

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Predict for frame, draw bounding box, and show
        results = input_model.predict(frame, conf=conf)
        annotated_frame = results[0].plot()
        cv2.imshow('Webcam YOLOv8', annotated_frame)

        # Exit the loop when ' ' is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()


# Fix YAML file paths
yaml_path = './manual_dataset/manual_data.yaml'
edit_yaml_paths(yaml_path)

# Load trained model
weights_path = './runs/detect/train6/weights/best.pt'
model = YOLO(weights_path)
print("Model successfully loaded")

# TRAIN AND VALIDATE MODEL ON MANUALLY LABELED DATASET
# results = model.train(data="./manual_dataset/manual_data.yaml",epochs=3)
# results = model.val()  # evaluate model performance on the validation set


#################################PREDICTION##########################################################

vid_path = "test_video.mp4"

"""
VIDEO ACTUAL TEXT:
Uh we didn't meet um until this morning. Um but I watched the France game when 
I got home. Um I didn't I didn't watch uh our whole game I watched the France game first.
"""

# loaded_video_predict = model.predict(vid_path, show=True, save_crop=False, conf=0.35)

webcam_predict(model, save_boxes=False, conf=0.25)
