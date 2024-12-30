# 1.Prepare the data

import os
import shutil
import random

# 80% of the dataset -> train folder
# 10% of the dataset -> validation folder
# 10% of the dataset -> test folder

#set path to the main dataset folder:
main_folder = 'E:\WoundProject\data_v2'

#set path for train, val, and test
train_folder = 'E:\WoundProject\data_v2-train'
val_folder = 'E:\WoundProject\data_v2-validation'
test_folder = 'E:\WoundProject\data_v2-test'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

#get the list of subfolders
subfolders = []

for f in os.scandir(main_folder):
    if f.is_dir():
        subfolders.append(f.path)

#split train / val / test
train_perc = 80
val_perc = 10

for subfolder in subfolders:

    # extract folder name
    subfolder_name = os.path.basename(subfolder)

    #create the train, validation, and test subfolder
    train_subfolder = os.path.join(train_folder , subfolder_name)
    val_subfolder = os.path.join(val_folder, subfolder_name)
    test_subfolder = os.path.join(test_folder , subfolder_name)

    os.makedirs(train_subfolder, exist_ok=True)
    os.makedirs(val_subfolder, exist_ok=True)
    os.makedirs(test_subfolder, exist_ok=True)

    #list all files in the current subfolder
    files = [f.path for f in os.scandir(subfolder) if f.is_file()]

    #shuffle the files
    random.shuffle(files)

    #calculate the number of train files
    num_train_files = int(len(files) * (train_perc / 100))
    num_val_files = int(len(files) * (val_perc / 100))
    print(num_train_files)

    #copy the files to the train folder
    for file in files[:num_train_files]:
        shutil.copy(file, os.path.join(train_subfolder, os.path.basename(file)))

    for file in files[num_train_files:num_train_files+num_val_files]:
        shutil.copy(file, os.path.join(val_subfolder, os.path.basename(file)))

    #copy the files to the test folder
    for file in files[num_train_files + num_val_files:]:
        shutil.copy(file, os.path.join(test_subfolder, os.path.basename(file)))

print("Finished copying the files to Train, Validation, and Test subfolder")

# 2. Build the Model

import tensorflow as tf
import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import efficientnet_v2
from keras.regularizers import l2

import numpy as np
import os
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

IMAGE_SIZE = 224 # all the images will be resized to this size

# get classes names
CLASSES = os.listdir(main_folder)
num_classes = len(CLASSES)

print(CLASSES)
print(num_classes)

# load the pre-trained model - EfficientNetV2 model
base_model = keras.applications.EfficientNetV2S(weights = 'imagenet', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3), include_top = False)
#include_top = False -> chop the last layer (1000 classes)

# all the layers will be fine-tuned during training
base_model.trainable = True

# for layer in base_model.layers[:-30]: #x
#     layer.trainable = False

#create a new model with more layers for our data

model = keras.Sequential([
    base_model,
    #layers.Flatten(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation = 'relu', kernel_regularizer= keras.regularizers.l2(7e-3)),
    layers.BatchNormalization(), #remove this if not successful
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer= keras.regularizers.l2(6e-3)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu', kernel_regularizer= keras.regularizers.l2(5e-3)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

from keras.optimizers import Adam
adam_opt = Adam(learning_rate=1e-4) # low value for transfer learning

model.compile(optimizer = adam_opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

#load the data
train_datagen = ImageDataGenerator(
    preprocessing_function=keras.applications.efficientnet_v2.preprocess_input, #x
    rescale=1./255,
    shear_range= 0.3,
    zoom_range=0.4,
    horizontal_flip=True,
    rotation_range=50,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=(0.9,1.1), #x
    channel_shift_range=0.2, #x
    fill_mode = 'nearest'
)

test_datagen = ImageDataGenerator(rescale= 1./255)

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=8,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

val_generator = test_datagen.flow_from_directory(
    val_folder,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=8,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=8,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

EPOCHS = 300
best_model_file = 'E:\WoundProject\data_v2-EfficientNetV2.keras'

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler

def lr_schedule(epoch, lr): #x
    if epoch < 10:
        return lr
    elif epoch < 70:
        return lr * 0.25
    else:
        return lr * 0.1

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_accuracy", patience=8,factor=0.4,verbose=1, min_lr=1e-8),
    #original: ReduceLROnPlateau(monitor="val_accuracy", patience=10,factor=0.1,verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_accuracy", patience=38,verbose=1, restore_best_weights=True),
    #original: EarlyStopping(monitor="val_accuracy", patience=25,verbose=1)
    #LearningRateScheduler(lr_schedule) #x
]

result = model.fit(
    train_generator, epochs = EPOCHS, validation_data= val_generator, callbacks = callbacks
)

# get the index of the epoch with the highest validation accuracy
best_val_acc_epoch = np.argmax(result.history['val_accuracy'])

# get the best validation accuracy value
best_val_acc = result.history['val_accuracy'][best_val_acc_epoch]

print("Best validation accuracy : " + str(best_val_acc))

# Plot the accuracy :

plt.figure(figsize=(10,6))
plt.plot(result.history['accuracy'],label = 'train acc')
plt.plot(result.history['val_accuracy'],label = 'val acc')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot the loss

plt.figure(figsize=(10,6))
plt.plot(result.history['loss'],label = 'train loss')
plt.plot(result.history['val_loss'],label = 'val loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:4f}")
print(f"Test Accuracy: {test_acc * 100:.4f}%")

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get tru labels and class names
true_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# Get predictions from the model
predictions = model.predict(test_generator)

# Calculate the confusion matrix
confusion = confusion_matrix(true_labels, np.argmax(predictions, axis = -1))

# Plot the confusion matrix
plt.figure(figsize=(20, 12))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()