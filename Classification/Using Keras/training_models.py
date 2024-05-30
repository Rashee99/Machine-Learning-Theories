import pickle
import time 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense 
from tensorflow.keras.callbacks import TensorBoard

Name = f'cat-vs-dog-prediction-{int(time.time())}'

tensorboard = TensorBoard(log_dir=f'logs\\{Name}\\')

x = pickle.load(open('X.pkl','rb'))
y = pickle.load(open('Y.pkl','rb'))

x = x/255

model = Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dense(128,input_shape = x.shape[1:], activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x,y, epochs=5, validation_split=0.1, batch_size=32, callbacks=tensorboard)