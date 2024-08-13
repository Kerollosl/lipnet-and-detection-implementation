# LipNet Lipreading Object Detection and video decoding model implementation

## Kerollos Lowandy

**Repository: lipnet-and-detection-implementation**

## GitHub Link
[https://github.com/Kerollosl/lipnet-and-detection-implementation](https://github.com/Kerollosl/lipnet-and-detection-implementation)

### Based on the following publication
[https://arxiv.org/pdf/1611.01599](https://arxiv.org/pdf/1611.01599)

### Tutorial Followed
[https://youtu.be/uKyojQjbx4c](https://youtu.be/uKyojQjbx4c)


### Necessary Packages
- imageio: 2.23.0
- tensorflow: 2.10.1
- ultralytics: 8.2.76
- cv2: 4.8.0.76
- pandas: 2.1.4
- matplotlib: 3.7.1
- gdown: 5.1.0
- roboflow: 1.1.37
- yaml: 6.0.2
- IPython: 7.34.0

### Directions

1. In a Python notebook environment, upload the `/runs/detect/train6/weights/best.pt` file to allow the model to begin training with pre-trained weights. Run the `train_roboflow_set.ipynb` notebook. Note: This process is done in a notebook to leverage simplified downloading of the Roboflow dataset and use of a virtual GPU.
2. Once the notebook has been run, download the created `runs.zip` file. Unzip and upload to the same directory as the `train_and_test_mouth_detection_manual_set.py` script. Navigate in the runs directory to find the path for the new `best.pt` file. Change the path in the `train_and_test_mouth_detection_manual_set.py` script on line 57 to match this path. 
3. Run the `train_and_test_mouth_detection_manual_set.py` script. This first trains the model on another dataset that manually had labels added through labelImg. The model is trained on this dataset as well as the Roboflow dataset because most of the data in the Roboflow dataset contains open mouths whereas most of the data in the manually labeled dataset contains closed mouths. To remove this extra training and proceed directly to prediction, comment out lines 62 and 63 of the script.
4. The `train_and_test_mouth_detection_manual_set.py` will then proceed to test the model's ability to detect a mouth through either live webcam prediction or through an uploaded video. The default setting is detecting for a live webcam. To change this, comment out line 78 and uncomment line 76. This will detect for `test_video.mp4`. To detect for another video, change line 68 to contain the path to the desired video.
5. After `train_and_test_mouth_detection_manual_set.py` has run to completion, find the file containing the newly generated `best.pt` file. This model will be used to detect the mouth in any video and extract an image of the bounding box in each frame to normalize and simplify the Lipnet model prediction process. Upload the file to the virtual contents of the `LipNet.ipynb` notebook. Then proceed to run this notebook. The notebook by default loads pre-trained weights for Lipreading prediction and does not perform any training. To further train the model with the pre-trained weights initially loaded, uncomment the line `#lipnet_model.fit(train, validation_data=test, epochs=50)` which is within the notebook block **Train Model**.
6. The model is tested on one of the homogenous train videos and a separate test video. To test the model on another desired video, upload the video to the notebook virtual contents and change the path within the line `predict_and_compare('/content/test_video.mp4', real_label=False)` to match the path of the desired video. Then change the **actual_test** string variable to match the word being spoken in the desired video


### Contents

- `train_and_test_mouth_detection_manual_set.py` - Program to train model with manually labeled dataset and test it with either live webcam video or an uploaded video.
- `train_roboflow_set.ipynb` - Program to download Roboflow dataset, load, train, and validate YOLOv8 mouth detection model, and save weights and validation metrics to a zip file to be downloaded. 
- `LipNet.ipynb` - Program used to preprocess LipNet dataset, create LipNet model architecture and load model weights, and predict words being said with any input video.
- `/runs/detect` - Subdirectory of YOLOv8 model training and validation containing the training weights and validation metrics.
- `/runs/detect/train6/weights/best.pt` - The current highest performing mouth detection model weights file from previous trainings. This model is used as part of the Lipnet model data preprocessing pipeline.
- `test_video` - A sample video to test both the mouth detection model and the Lipnet model. 
- `/manual_dataset/` - A dataset containing over 7,000 images of people with a variety of different mouth positions. About 100 of these images have been labeled and placed in the `/Human_Photos_Labeled/` subdirectory. The rest remain in the `/Humans/` subdirectory. For further model training and performance increase, more of the unlabeled images can be labeled using labelImg and transferred to the `/Human_Photos_Labeled/` directory. 