from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import winsound

model = load_model('cifar100.h5')
labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

cam = cv2.VideoCapture(0)

cv2.namedWindow("Video")

import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:  
   print("Please wait. Calibrating microphone...")  
   # listen for 3 seconds and create the ambient noise energy level  
   r.adjust_for_ambient_noise(source, duration=3)  
   print("Say something!")  
   audio = r.listen(source, timeout=3)  

try:
    word = r.recognize_google(audio)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
print('Processing audio')
try:
    num = labels.index(word)
except ValueError:
    print(word + " Not found")

print(labels[num])
while True:
    ret, frame = cam.read()
    cv2.imshow("Video", frame)
    
    image = frame
    m = image.shape[0]//3
    n = image.shape[1]//3
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    abort = 0
    
    for i in range(0,image.shape[1]+1,n):
        for j in range(0,image.shape[0]+1,m):
            for k in range(i+n,image.shape[1]+1,n):
                for l in range(j+m,image.shape[0]+1,m):
                    print(i,j,k,l)
                    im = image[j:l,i:k]
                                   
                    res = cv2.resize(im,(32, 32), interpolation = cv2.INTER_AREA)  
                    res = np.resize(res,(1,32,32,3))
                    res = res.astype('float32')
                    res = res/255
                    x = model.predict_proba(res)[0]
                    x = process(x)
                    
                    if(x[num] > 0.55):
                        frequency = 1500
                        duration = 200
                        winsound.Beep(frequency, duration)
                        abort = 1

                        print(labels[num])
                        break
                if abort:
                    break
            if abort:
                break
        if abort:
            break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

            
cam.release()

cv2.destroyAllWindows()