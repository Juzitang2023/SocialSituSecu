﻿from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

### 超参数：
# dimensions of InceptionV3.
img_width, img_height = 224, 224 #

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 15000
nb_validation_samples = 8286
epochs = 20
batch_size = 32


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# 构建不带分类器的预训练模型
base_model = InceptionV3(weights='imagenet', include_top=False)

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加一个全连接层
x = Dense(512, activation='relu')(x)

# 添加一个分类器，假设我们有3个类
predictions = Dense(3, activation='softmax')(x)

# 构建我们需要训练的完整模型
model = Model(inputs=base_model.input, outputs=predictions)

# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 InceptionV3 的卷积层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型（一定要在锁层以后操作）
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=["accuracy"])


### 数据集
## 扩充数据集
# 这是我们将用于训练的扩充配置
train_datagen = ImageDataGenerator(rescale=1. / 255, # 预测的时候，特征也要这么处理
    shear_range=0.2, # 用来进行剪切变换的程度
    zoom_range=0.2, # 用来进行随机的放大
    horizontal_flip=True) # 随机的对图片进行水平翻转
# 这是我们将用于测试的扩充配置：
valid_datagen = ImageDataGenerator(rescale=1. / 255)# 仅缩放

# 这是一个生成器，将读取在“train_data_dir”子文档中找到的图片，并无限期地生成一批增强图像数据
train_generator = train_datagen.flow_from_directory( # 使用.flow_from_directory()来从我们的jpgs图片中直接产生数据和标签。
    train_data_dir,
    target_size=(img_width, img_height), # 所有图像将调整为224x224
    batch_size=batch_size,
    class_mode='categorical', 
    # classes = ["political","porn","terrorism"]
)
print(train_generator.class_indices) #{'political': 0, 'porn': 1, 'terrorism': 2}


validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
)
print(validation_generator.class_indices) # 输出生成的label名称 { 'political': 0, 'porn': 1, 'terrorism': 2}

### 在新的数据集上训练几代
print("开始训练：")
model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size, # 一个 epoch 完成并开始下一个 epoch 之前,如果未指定，将使用len(generator) 作为步数。
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

## 保存模型
model.save('data/modelFile/my_model.h5')  # 创建 HDF5 文件 'my_model01.h5'
print("保存模型成功。")


