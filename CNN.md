---

---

# CNN

convolution filter 와 max pooling을 거쳐 DNN 을 적용하는것

기본적으로 DNN보다 성능이 좋다.

그래서 DNN을 안쓰고 CNN을 쓴다 (영상에든 어디에든)

DNN으로 되는것이 CNN으로 다되기 때문



(요새 Tensorflow를 잘 안쓰고 Keras를 쓰는 이유와 같은 것 같다)

CNN의 구조를 어떻게 하느냐에 따라 RNN,GAN 로 분류한다고 할 수 있다.

Deeplearning's Father = CNN 

Deeplearning하면 CNN이라고 봐도 무방.







CNN process

![CNN](CNN.assets/2.png)



그림와 같이 CNN은 밑의 2개의 부분으로 구성된다.

#### feature extraction => conv layer + pooling layer

#### classification => fully connected layer

그 후 DNN의 과정을 지나 출력이 된다.

데이터의 구성에 따라 Convolution Filter 값을 어떻게 줄 것인지

Maxpooling 값을 어떻게 줄것인지는 아직 정형화 된 것이 없다.



속도가 느리다면 텐서플로 버전을 낮추어 보는방법도 시도하라

```python

# eye_classification.ipynb 영상처리 실습
#실제 영상분류할때 쓰는 코드이므로 알아두자

import os

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import VGG16


conv_layers = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
conv_layers.trainable = False

model = models.Sequential()

model.add(conv_layers)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer="RMSprop", metrics=['acc'])


data_aug_generator = ImageDataGenerator(
      rotation_range=10,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True,
      vertical_flip=False,
      preprocessing_function=vgg16.preprocess_input
)

data_no_aug_generator = ImageDataGenerator(
      preprocessing_function=vgg16.preprocess_input
)

train_data_generator = data_aug_generator.flow_from_directory(
      "train",
      target_size=(224,224),
      batch_size=64,
      class_mode='sparse'
)

test_data_generator = data_no_aug_generator.flow_from_directory(
      "test",
      target_size=(224,224),
      class_mode='sparse'
)


model.fit_generator(
      train_data_generator,
      validation_data=test_data_generator,
      validation_steps=5,
      steps_per_epoch=train_data_generator.samples/64,
      epochs=10
)


y_ = model.predict_generator(
      test_data_generator,
      steps=test_data_generator.samples/64
)

custom_labels = list(test_data_generator.class_indices.keys())
predicted = np.argmax(y_, axis=1)
print(predicted[0], custom_labels[predicted[0]])
```



# Deep Learning libaray process

1.연구 + Try

2.Try + 결과

3.모델 사용 시작

4.패키징 + 일반화



# Etc

test loss값이 증가하는 구간이 overfitting이 발생하기 시작하는 것