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

#for layer in base_model.layers[:-200]: #x
    #layer.trainable = False

#create a new model with more layers for our data

embedding_layer = layers.Dense(128, activation='relu', name="embedding_layer")

model = keras.Sequential([
  base_model,
  layers.GlobalAveragePooling2D(),
  layers.Dense(1024, activation = 'relu', kernel_regularizer= keras.regularizers.l2(2e-3)),
  layers.BatchNormalization(), #remove this if not successful
  layers.Dropout(0.4),
  layers.Dense(512, activation='relu', kernel_regularizer= keras.regularizers.l2(2e-3)),
  layers.BatchNormalization(),
  layers.Dropout(0.3),
  layers.Dense(256, activation='relu', kernel_regularizer= keras.regularizers.l2(2e-3)),
  layers.BatchNormalization(),
  layers.Dropout(0.2),
  layers.Dense(128, activation='relu', kernel_regularizer= keras.regularizers.l2(2e-3)),
  layers.BatchNormalization(),
  layers.Dropout(0.1),
  embedding_layer,
  layers.Dense(num_classes, activation='softmax')
])

from keras.optimizers import Adam
from keras import backend as K
from tensorflow.keras.optimizers.schedules import CosineDecay

def focal_loss(alpha=0.25, gamma=2.0, smoothing=0.025):
    def loss(y_true, y_pred):
        y_true = y_true * (1-smoothing) + smoothing / num_classes
        y_pred = K.clip(y_pred, 1e-9, 1.0)
        ce = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        fl = ce * weight
        #loss = -y_true * (alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred))
        return K.sum(fl)
    return loss

# Define the Cosine Decay schedule
initial_lr = 1e-4
decay_steps = 1000
alpha = 1e-6

cosine_decay = CosineDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps,alpha=alpha)

adam_opt = Adam(learning_rate=1e-4) # low value for transfer learning
#adam_opt = Adam(learning_rate=cosine_decay)

model.compile(optimizer = adam_opt, loss = focal_loss(), metrics = ['accuracy'])
#model.compile(optimizer = adam_opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

#load the data
train_datagen = ImageDataGenerator(
  preprocessing_function=keras.applications.efficientnet_v2.preprocess_input, #x
  rescale=1./255,
  shear_range= 0.225,
  zoom_range=0.275,
  horizontal_flip=True,
  vertical_flip=True,
  rotation_range=55,
  width_shift_range=0.225,
  height_shift_range=0.225,
  brightness_range=(0.925,1.075),
  channel_shift_range=0.125,
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

EPOCHS = 750
log_dir = 'E:/WoundProject/logs'
best_model_file = 'E:\WoundProject\data_v2-EfficientNetV2.keras'

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [
  ModelCheckpoint(best_model_file, verbose=1, save_best_only=True, monitor="val_accuracy"),
  ReduceLROnPlateau(monitor="val_accuracy", patience=6,factor=0.67,verbose=1, min_lr=1e-9),
  EarlyStopping(monitor="val_accuracy", patience=20,verbose=1, restore_best_weights=True),
  tensorboard_callback
]

from sklearn.utils.class_weight import compute_class_weight

# Get class indices from the train generator
class_indices = train_generator.class_indices

# Convert class indices to a sorted list of class labels
class_labels = list(class_indices.keys())

# Get class counts from the generator
class_counts = train_generator.classes

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_counts),y=class_counts)

# Convert to dictionary format
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

print("Class Weights:", class_weights_dict)

result = model.fit(
  train_generator, epochs = EPOCHS, validation_data= val_generator, callbacks = callbacks, class_weight= class_weights_dict
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

from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Evaluate test accuracy
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.4f}%")

# Get tru labels and class names
true_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# Get predictions from the model
predictions = model.predict(test_generator)

# Calculate the confusion matrix
confusion = confusion_matrix(true_labels, np.argmax(predictions, axis = -1))

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion / np.sum(confusion, axis=1)[:, np.newaxis], annot=True, fmt=".2%", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Get predictions from the model
y_pred = np.argmax(predictions, axis = -1)

# Print the classification report
print("Classification Report:\n")
print(classification_report(true_labels, y_pred, target_names=class_names, zero_division=1))

from tensorflow.keras.models import Model
from tensorboard.plugins import projector

# Create model for embeddings
embedding_model = Model(inputs=model.input, outputs=model.get_layer("embedding_layer").output)

# Get file paths of test images
test_file_paths = []
test_labels = []

for i in range(len(test_generator.filenames)):
    test_file_paths.append(os.path.join(test_folder, test_generator.filenames[i]))
    test_labels.append(test_generator.classes[i])

# Read and preprocess images
import cv2
X_test_images = []

for img_path in test_file_paths:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))  # Resize
    img = img / 255.0  # Normalize
    X_test_images.append(img)

X_test_images = np.array(X_test_images)

# Generate embeddings manually
embeddings = embedding_model.predict(X_test_images)

labels = test_generator.classes

# Create a checkpoint directory
checkpoint_dir = os.path.join(log_dir, "checkpoint")
os.makedirs(checkpoint_dir, exist_ok=True)

# Save model checkpoint
checkpoint_path = os.path.join(checkpoint_dir, "model")
model.save(checkpoint_path)

# Save embeddings & metadata
np.savetxt(os.path.join(log_dir, 'embeddings.tsv'), embeddings, delimiter='\t')

# Save class labels for metadata
class_names = list(test_generator.class_indices.keys())
metadata_path = os.path.join(log_dir, 'metadata.tsv')

with open(metadata_path, 'w') as f:
    for label in test_labels:  # Ensure labels are correctly mapped
        f.write(f'{class_names[label]}\n')

# Ensure correct tensor name
embedding_tensor_name = embedding_model.get_layer("embedding_layer").output.name.split(":")[0]

# Configure TensorBoard embedding visualization
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_tensor_name
embedding.metadata_path = "metadata.tsv"

# Save the projector config file
with open(os.path.join(log_dir, "projector_config.pbtxt"), "w") as f:
    f.write(str(config))

print(f"Embeddings and metadata saved in {log_dir}")
print(embedding_model.get_layer("embedding_layer").output.name)