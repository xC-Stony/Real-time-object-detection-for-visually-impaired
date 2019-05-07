
#(X_train, y_train), (X_test, y_test) = cifar100.load_data()

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from keras.datasets import cifar100


model = load_model('cifar100.h5')



labels = [
    'apple', 'aquarium_fish', 'person', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'person', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'chair', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'person', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'person', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'keyboard', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'person',
    'worm'
]



for j in range(1,14):
    image = cv2.imread("{}.jpg".format(j))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

        
    res = cv2.resize(image,(32, 32), interpolation = cv2.INTER_AREA)    
    res = np.resize(res,(1,32,32,3))
    res = res.astype('float32')
    res = res/255
    
    x = model.predict_proba(res)[0]
    z = np.argmax(x)
    print(labels[z])    
    time.sleep(1)

#j=117
#image = cv2.imread("im{}.jpg".format(j))
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#m = image.shape[0]//3
#n = image.shape[1]//3
#
#
#res = cv2.resize(image,(32, 32), interpolation = cv2.INTER_AREA)  
#res = np.resize(res,(1,32,32,3))
#res = res.astype('float32')
#res = res/255
#
#
#
#x = model.predict_proba(res)[0]
#ind = np.argpartition(x, -5)[-5:]
#for i in ind:
#    print(labels[i],x[i])  
#    
#
#for i in range(0,image.shape[1]+1,n):
#    for j in range(0,image.shape[0]+1,m):
#        for k in range(i+n,image.shape[1]+1,n):
#            for l in range(j+m,image.shape[0]+1,m):
#                print(i,j,k,l)
#                im = image[j:l,i:k]
#                               
#                res = cv2.resize(im,(32, 32), interpolation = cv2.INTER_AREA)  
#                res = np.resize(res,(1,32,32,3))
#                res = res.astype('float32')
#                res = res/255
#                x = model.predict_proba(res)[0]
#                x[2] = x[46] +x[2] +x[11]+x[98] +x[35]
#                x[[46,11,98,35]] = 0   
#                x[9] = x[9] + x[16]
#                p = 0
#                for ix in range(0,100):
#                    if x[ix] > 0.4:
#                        if p == 0:
#                            plt.imshow(im)
#                            plt.show()                           
#                            p = 1
#                        print(labels[ix],x[ix])
