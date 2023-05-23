import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from keras.optimizers import SGD, RMSprop
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils, to_categorical, load_img, img_to_array
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LeakyReLU
from keras.utils import load_img, img_to_array

from sklearn import preprocessing


vid = cv2.VideoCapture(0)
print("Camera connection successfully established")
i = 0
model = Sequential()


new_model = load_model('bai4.h5')

while(True):
    r, frame = vid.read()
    cv2.imshow('frame', frame)
    cv2.imwrite('C:\\Users\\DELL\\Documents\\AI_Cuoiki\\test'+ str(i) + ".jpg", frame)
    test_image = load_img('C:\\Users\\DELL\\Documents\\AI_Cuoiki\\test' + str(i) + ".jpg", target_size=(150, 150))
    test_image = img_to_array(test_image)
    test_image=test_image.astype('float32')
    test_image = np.expand_dims(test_image, axis=0)
    result = (new_model.predict(test_image).argmax())
    classes = ['hoa_lan','hoa_sen','hoadao','hoahong','hoahuongduong','hoamai','hoasu']

    print('Đây là : {}'.format(classes[result]))
    os.remove('C:\\Users\\DELL\\Documents\\AI_Cuoiki\\test' + str(i) + ".jpg")
    i = i + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()