# **Pedestrian-detection**

#### **Using python 2.7 and opencv 3**

*Welcome to my repository of Pedestrian Detection. We will use #INRIA, #SVC, #HoG, #NMS in this project.*

```All the rights of this project are for Hesam Korki, and a Special thanks to Dr.Nasihatkon from KNTU```

>*Contact me: Hesam.korki@gmail.com*

*Classified with the score of 0.95* 


## **Goals**

- *Compute HoG features from images of a given data set, and extract image patches (both positive and negative examples).*

- *Train an SVM classifier to perform classification and detection tasks.*

## **DATASET**

First things first in this project we will use the INRIA person dataset for both training and testing, but after training our classifier we can test it on other pictures to. You need to download the dataset from its official website here: 

>[INRIA PERSON DATASET](http://pascal.inrialpes.fr/data/human/)

1-The positive images are extracted from the original images and located (in two different sizes) in directories starting with the patch size (96X160H96, etc.).
2- According to the webpage, you should only use the 64x128 central part of these images. The reason why the pictures have been widened in both width and height is brought on the page.
3-Dividing the dataset to Train and Test are up to you. Although dataset providers seem to have fixed this, you are free to choose any portion of the dataset for training and the rest for testing.
4-Negative images are not provided in 64x128 patches. Read the documentation for details on how to build negative patches from negative images provided. Building the negative data is all up to you.
**_We will get random 64*128 windows for the negative samples._**

## **HoG Descriptor**

You can find the official documentation here:

>[HoG Descriptor](https://docs.opencv.org/3.4.1/d5/d33/structcv_1_1HOGDescriptor.html)

In order to set the conductor parameter of HoGDescriptor, we will create an XML file.
After reading images and extracting their features with the `HoG.compute()` class, append them to an empty list and give the label of 1 to those including pedestrian in them and 0 to those not including pedestrians.

## **Training our SVM**

In this project, the goal is to use a custom SVM classifier and not ~~the cv2.HOGDescriptor_getDefaultPeopleDetector()~~, and that is the meaning of training.
Now we have to train an SVM classifier. OpenCV provides its implementation of SVM. But since OpenCV’s SVM is not properly documented, we will be using the `SVC (support vector classifier)` class in the scikit-learn library, a very popular machine learning package. Find the documentation here:

>[SVC from scikit.svm](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

### **Pickle Trick** 


Since the process of training involves reading up to 5000 pictures and extracting their features, it will take time and of course something about 1.5 GB of your RAM. So we will train our classifier only once and save it with the pickle library and load the file for other times.

## **SetSVMDetector**

Now that we have our support vector, we can easily set our svmdetector using the class 'HOG.setSVMDetector'.

## **Testing and drawing a box around pedestrians** 

We will use class `HOG.detectMultiScale()` in order to get a moveable window through our test image and different scales of our test image. The whole information you need to deal with to comprehend its parameters and functionality is mentioned here: 

>[HoG.Multiscale() parameters](https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/)

*Had you have any further questions, you are welcome to ask me* 

>Email: Hesam.korki@gmail.com

*Hope that you find this helpful.
Best Regards Hesam Korki*
