import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

def resize(image, size):
    img = tf.io.read_file(image)
    x = tf.keras.preprocessing.image.smart_resize(tf.image.decode_jpeg(img, channels=3), [size, size],
                                               interpolation='bilinear')
    tf.keras.preprocessing.image.save_img(image, x)

def match_images(name):

    template = cv2.imread('Template_Matching/templates/v6_228_T_v2.jpg')
    #template = cv2.imread('templates/v6_228_T_v2.jpg')
    original_image = cv2.imread(name)

    original_copy = original_image.copy()

    original_copy = cv2.cvtColor(np.array(original_copy), cv2.COLOR_BGR2GRAY)

    blur_t = cv2.blur(template,(5,5))
    blur = cv2.blur(original_copy,(5,5))
    template = cv2.Canny(blur_t, 125, 225)
    img = cv2.Canny(blur, 50, 100)

    #cv2.imwrite('Data/Template_Matching_Results/ex1/Template_canny.jpg', template)
    w, h = template.shape[::-1]
    detected = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(detected)

    #print (f'min_val:{min_val}, max_val:{max_val}, min_loc:{min_loc}, max_loc:{max_loc}')
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(original_image, top_left, bottom_right, 255, 2)

    x1, y1 = top_left
    x2, y2 = bottom_right
    # removing blue frame form image
    x1 = x1 + 2
    x2 = x2 - 2
    y1 = y1 + 2
    y2 = y2 - 2

    image = original_image[y1:y2, x1:x2]
    #image = original_copy[y1:y2, x1:x2]
    cv2.imwrite(name, image)
    #return image

def preprocess_images():
    df = pd.read_csv('Train.csv')
    for index, row in df.iterrows():
        p = row['Path']
        if (row['Frontal/Lateral'] == 'Frontal'):
            resize(p, 250)
            match_images(p)
        elif (row['Frontal/Lateral'] == 'Lateral'):
            resize(p, 224)

    df2 = pd.read_csv('Valid.csv')
    for index, row in df2.iterrows():
        p = row['Path']
        if (row['Frontal/Lateral'] == 'Frontal'):
            resize(p, 250)
            match_images(p)
        elif (row['Frontal/Lateral'] == 'Lateral'):
            resize(p, 224)
