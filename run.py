import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

cancers = ['malignant', 'benign']

model = load_model('CancerModel-5.h5')

for file in os.listdir('cancers'):
  img = load_img('cancers/' + file, target_size=(224, 224))
  img = img_to_array(img)
  img = [list(img)]
  img = np.array(img)
  pred = model.predict(img)[0]
  pred = cancers[np.argmax(pred)]

  cancer = file.replace('.jpg', '').replace('0', '').replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '')

  if pred != cancer:
    pred += ' [Error]'

  print(file, ':', cancer, pred)
