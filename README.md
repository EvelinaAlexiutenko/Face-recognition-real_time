# Face-recognition-real_time
Using FaceNet, MTCNN and SVM to solve face recognition problem in real-time.
<img src="https://user-images.githubusercontent.com/58363847/167313443-9e2c1da3-5b93-46af-a943-2a029c81c846.jpg" width="500" height="400" /><br>
To see the video of webcam real-time detection click:
<a href="https://youtu.be/2NHvwepAyAk">Face recognition with FaceNet</a><br>
Or to see how does it work if some part of the face is hidden:
<a href="https://youtu.be/MV_SIRYCpy0">Face recognition in details</a>
<br>
In repo you can find file Face_recognition.ipynb(functional approach) and use it for quick result. Or you can choose FaceTrainer.py and FaceDetector.py(OOP approach).
## Requirements:
cuda_11.2 <br>
python version- 3.6.2
<br>
## Algorithm:
<details>
  <summary><em>Data preparation</em></summary>
  You can make your custom dataset with the structure - one subdirectory for each person: <br>
  <img src="https://user-images.githubusercontent.com/58363847/167355965-eb791e52-a8d9-4381-8c63-719b12fbe5bd.jpg" wight = 400 height =400/><br>
  By working with this project FaceNet badly recognized people on selfies that's why I need to do augmentation. The augmentation function resizes the original photo
  and overlays it on a bigger background.
</details>
<details>
  <summary><em>Detection of the face with MTCNN</em></summary>
  I am using this network to extract face from the photo. If MTCNN does not find a face, please go back to the previous step of the algorithm and do augmentation.
  Example of MTCNN application:
  <p float="left">
  <img src="https://user-images.githubusercontent.com/58363847/167314019-abf4ee59-ce82-4870-8308-1c6f80e73938.jpg"/>
  </p>
  MTCNN also find a person in a sunglasses:<br>
  <img src="https://user-images.githubusercontent.com/58363847/167359791-1d6622c7-19df-4426-b56f-ff5bcf2f2987.png"/><br>
  </details>
<details>
  <summary><em>Creating an embedings with FaceNet</em></summary>
  I use pretrained FaceNet because of the fact that recognition NN need to be train on a large dataset. You can find and download model
  <a href="https://github.com/nyoki-mtl/keras-facenet">here</a>. We need embedings to perform vector classification. The FaceNet model can be used as part of the
  classifier itself, or we can use the FaceNet model to pre-process a face to create a face embedding that can be stored and used as input to our classifier model.
  This latter approach is preferred as the FaceNet model is both large and slow to create a face embedding.
  </details>
  <details>
  <summary><em>Perform Face Classification with SVM</em></summary>
  We will use the SVM classifier model to predict by embedding the identity of a given face.<br>
  Testing on validation sample:
  <br>
  <img src="https://user-images.githubusercontent.com/58363847/167361413-f0a186a2-969a-44e1-bb61-879f8ec76246.jpg" wight = 300 height =300/><br>
  </details>
  <details>
  <summary><em>Real-time webcam face recognition</em></summary>
  
  </details>
<details>
  <summary><em>References</em></summary>
  
  1. https://public.roboflow.com/object-detection/uno-cards - Uno cards dataset;
  
  2. https://github.com/ultralytics/yolov5 - Original repo of YOLOv5;
  
  3. https://models.roboflow.com/ - Model zoo from roboflow;
  
  4. https://www.youtube.com/watch?v=nDPWywWRIRo&t=3256s&ab_channel=StanfordUniversitySchoolofEngineering - Basic Object Detection knowledge;
  
  5. https://www.youtube.com/watch?v=MdF6x6ZmLAY&t=1508s - Yolov5 tutorial;
  
  6. https://www.youtube.com/watch?v=NU9Xr_NYslo&t=607s - Yolov5 tutorial;
 
  7. https://www.youtube.com/watch?v=yfDjsuxIKA4&t=2718s - Training other models using Tensorflow Object Detection;
  
  8. https://www.youtube.com/watch?v=pnntrewH0xg&t=151s - Example of web-app for testing your model;
  
  9. https://www.youtube.com/watch?v=TB-fdISzpHQ&t=3717s - Another Basic Object Detection knowledge;
  
  10. https://towardsdatascience.com/yolo-v4-or-yolo-v5-or-pp-yolo-dad8e40f7109 - Difference between the last YOLO-type models;
  
  11. https://techzizou.com/category/object-detection/ - Web app on tf2;
  
  12. https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md - model zoo(tf2);
  </ul>
</details>
<br>

