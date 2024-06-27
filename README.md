# Project Name: # Real-Time-Face-Emotion-Recognition-System

Table of Contents:
1.[Description](#p1)

2.[Installations](#p2)

3.[Usage](#p3)

4.[Dataset](#p4)



![](https://github.com/omar178/Emotion-recognition/blob/master/emotions/Happy.PNG)
![](https://github.com/omar178/Emotion-recognition/blob/master/emotions/angry.PNG)




<a id="p1"></a> 
# Description:

Our human face has mixed emotions, so we are to demonstrate the probabilities of these emotions that we have.

## What does emotion recognition mean?

Emotion recognition is a technique used in software that allows a programme to "read" the emotions on a human face using advanced image processing. Companies have been experimenting with combining sophisticated algorithms with image processing techniques that have emerged in the past ten years to understand more about what an image or a video of a person's face tells us about how he or she is feeling, and not just that but also showing the probabilities of mixed emotions a face could have.

<a id="p2"></a> 
# Installations:
-keras

-imutils

-cv2

-numpy

<a id="p3"></a> 
# Usage:

The programme will create a window to display the scene captured by the webcamera and a window representing the probabilities of detected emotions.

> Demo

python real_time_video.py

You can just use this with the provided pretrained model I have included in the path written in the code file. I have chosen this specifically since it scores the best accuracy. Feel free to choose any, but in this case, you have to run the later file. train_emotion_classifier
If you just want to run this demo, the following content can be skipped:
- Train

python train_emotion_classifier.py


<a id="p4"></a> 
# Dataset:

I have used this (https://www.kaggle.com/c/3364/download-all) dataset.

Download it and put the CV in FER2013/FER2013/.

2013 emotion classification test accuracy: 66%


# Credits
This work is inspired by [this] (https://github.com/oarriaga/face_classification). Great work, and the resources of Adrian Rosebrock helped me a lot! .

# Ongoing 
Draw emotional faces next to the detected face.

Issues and Suggestions

If you have any issues or suggestions for me, you can create an issue (https://github.com/omar178/Emotion-Recognition/issues).

If you like this work, please help me by giving me some stars.
