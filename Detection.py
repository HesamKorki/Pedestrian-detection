## READ THE README FILE 
## All the rights of this project are for Hesam Korki
## Special thanks to Dr.Nasihatkon
## Contact me: Hesam.korki@gmail.com

import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
from sklearn.svm import *
import pickle
import itertools
import glob
import os.path

model= 'svm_eq.pkl'

# HoG descriptor

hog = cv2.HOGDescriptor('hog.xml')


image_features = []
target = []

print "Computing positive HoG features"
for fname in glob.glob('INRIAPerson/96X160H96/Train/pos/*'):
    I = cv2.imread(fname)
    
    I = I[16:-16,16:-16]

    U = hog.compute(I)


    image_features.append(U)
  
    target.append(1)


print "Computing negative HoG features"
for fname in glob.glob('INRIAPerson/Train/neg/*'):
    I =cv2.imread(fname)
    sample2 = np.random.randint(0,I.shape[0]-128,10)
    sample1 = np.random.randint(0,I.shape[1]-64,10)
    samples = zip(sample1,sample2)
    for j in samples:
        I1 = I[j[1]:j[1]+128,j[0]:j[0]+64,:]
        U = hog.compute(I1)
        image_features.append(U)
        target.append(0)


print "Training the svm"
X = np.asarray(image_features,dtype=np.float64)
Y = np.asarray(target,dtype= np.float64)
X= np.reshape(X, (X.shape[0],X.shape[1]))

## Pickle Trick

print 'Saving the classifier'

if os.path.isfile(model)== True:
    svm = pickle.load(open(model,'rb'))
    pass
else:
    svm = SVC(kernel='sigmoid', gamma='auto', C= 5.0, max_iter=-1, tol=1e-4, coef0=1)
    svm.fit(X, Y)
    pickle.dump(svm, open(model, 'wb'))




support_vectors = []

support_vectors.append(np.dot(svm.dual_coef_,svm.support_vectors_)[0])

support_vectors.append([svm.intercept_])

support_vectors = list(itertools.chain(*support_vectors))

hog.setSVMDetector(np.array(support_vectors,dtype=np.float64))

#---------------------------- Get the Score of Test data
# Uncomment this part if you want to get a score from a test data on your classifier

# If_test = []
# target_test = []
# for fname in glob.glob('INRIAPerson/70X134H96/Test/pos/*'):
#     I = cv2.imread(fname)
#     I = I[3:-3,3:-3]
#     U = hog.compute(I)
#     If_test.append(U)
#     target_test.append(1)
#
# for fname in glob.glob('INRIAPerson/Test/neg/*'):
#     I =cv2.imread(fname)
#     sample2 = np.random.randint(0,I.shape[0]-128,1)
#     sample1 = np.random.randint(0,I.shape[1]-64,1)
#     samples = zip(sample1,sample2)
#     for j in samples:
#         I1 = I[j[1]:j[1]+128,j[0]:j[0]+64,:]
#         U = hog.compute(I1)
#         If_test.append(U)
#         target_test.append(0)
#
# N = np.asarray(If_test,dtype=np.float64)
# M = np.asarray(target_test,dtype= np.float64)
# N= np.reshape(N, (N.shape[0],N.shape[1]))
#
# print svm.score(N,M)
# #--------------------------------------------------



for fname in glob.glob('Test/*'):
    Itest = cv2.imread(fname)
    Itest = imutils.resize(Itest,width=min(400,Itest.shape[1]))
    #Itest = Itest[16:-16,16:-16,:]
    #Ty = cv2.resize(Itest, (64, 128), interpolation=cv2.INTER_AREA)
    #G = cv2.cvtColor(Ty, cv2.COLOR_BGR2GRAY)
    o = Itest.copy()
    # detect people in the image
    rects, weights = hog.detectMultiScale(Itest, winStride=(1,2),scale=1.4 , padding=(8,8),
                                          finalThreshold=100 , useMeanshiftGrouping=0)

    for (x, y, w, h) in rects:
        cv2.rectangle(o, (x, y), (x + w, y + h), (0, 255, 255),2)


    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)


    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(Itest, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow("Before NMS", o)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    cv2.imshow("After NMS", Itest)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break






cv2.destroyAllWindows()
