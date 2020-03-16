import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential

print('getting data')

train_size = 1150
test_size = 300

xtrain, ytrain = [], []
xtrain += ['images/train/malignant/' + image for image in os.listdir('images/train/malignant')[:train_size]]
ytrain += [[1, 0] for i in range(train_size)]
xtrain += ['images/train/benign/' + image for image in os.listdir('images/train/benign')[:train_size]]
ytrain += [[0, 1] for i in range(train_size)]

i = np.arange(len(xtrain))
np.random.shuffle(i)
xtrain, ytrain = np.array(xtrain)[i], np.array(ytrain)[i]

xtest, ytest = [], []
xtest += ['images/test/malignant/' + image for image in os.listdir('images/test/malignant')[:train_size]]
ytest += [[1, 0] for i in range(train_size)]
xtest += ['images/test/benign/' + image for image in os.listdir('images/test/benign')[:train_size]]
ytest += [[0, 1] for i in range(train_size)]

i = np.arange(len(xtest))
np.random.shuffle(i)
xtest, ytest = np.array(xtest)[i], np.array(ytest)[i]

print('got data')

class dataGen(Sequence):
  def __init__(self, x, y, batch_size):
    self.x, self.y = x, y
    self.batch_size = batch_size

  def __len__(self):
    return int(len(self.x) / self.batch_size)

  def __getitem__(self, idx):
    x = [list(img_to_array(load_img(path, target_size=(224,224)))) for path in self.x[idx*self.batch_size : (idx+1)*self.batch_size]]
    y = self.y[idx*self.batch_size : (idx+1)*self.batch_size]

    return np.array(x), np.array(y), [None]

train = dataGen(xtrain, ytrain, 50)
train_steps = len(train)
test = dataGen(xtest, ytest, 50)
test_steps = len(test)
del xtest, ytest, xtrain, ytrain

model = Sequential()
model.add(Conv2D(10, (3, 3), activation='elu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.05))
model.add(Conv2D(10, (3, 3), activation='elu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.05))
model.add(Conv2D(10, (3, 3), activation='elu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.05))
model.add(Conv2D(5, (3, 3), activation='elu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.05))
model.add(Conv2D(5, (3, 3), activation='elu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.05))
model.add(Flatten())
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.05))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('training')

for i in range(10):
    model.fit_generator(generator=train, steps_per_epoch=train_steps, epochs=1)
    print(model.evaluate_generator(generator=test, steps=test_steps))
    model.save('CancerModel.h5')
