from keras.engine import Model
from keras.layers import *
from keras.applications import *

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Data
# X = input images [num, row, cols, channel]
# y = input controls 2D matrix [throttle value, steering angle]
X, y, X_test, y_test = None, None, None, None

# resolution for images (row, col, channel)
X=np.load(open('/home/student/Documents/dixant/training_data/1.img/X1.npy','rb'))
image_dims = X.shape[1:]

# get pre-trained model
xception = ResNet50(include_top=False,
              weights='imagenet',
              input_shape=image_dims)

# add last layers
layer = Flatten()(xception.output)
layer = Dense(1024, activation='relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(1024, activation='relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(1024, activation='relu')(layer)
layer = Dropout(0.5)(layer)

# output controls. it will be Linear function
output_layer = Dense(2)(layer)

# compile the model
model = Model(xception.input, output_layer)
model.load_weights('model/trained_model.h5')
model.compile(optimizer='adam', loss='mse')

print('Model loaded!')

# train for 6 data batches

for i in range(6):
    X=np.load(open('/home/student/Documents/dixant/training_data/1.img/X'+str(i)+'.npy','rb'))
    y=np.load(open('/home/student/Documents/dixant/training_data/1.img/y'+str(i)+'.npy','rb'))
    print('Data file loaded: ',i)
    model.fit(x=X, y=y, epochs=1,batch_size=2) 
    model.save('model/trained_model.h5')

X_test=np.load(open('/home/student/Documents/dixant/training_data/1.img/X6.npy','rb'))
y_test=np.load(open('/home/student/Documents/dixant/training_data/1.img/y6.npy','rb'))

print('Test loss: ', model.evaluate(x=X_test,y=y_test,batch_size=2))