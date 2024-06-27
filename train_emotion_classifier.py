"""
Description: Train emotion classification model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from load_and_process import load_fer2013, preprocess_input
from cnn import mini_XCEPTION
from sklearn.model_selection import train_test_split
import warnings

# Suppress specific Keras warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

# Parameters
batch_size = 32
num_epochs = 10000
input_shape = (48, 48, 1)
validation_split = 0.2
verbose = 1
num_classes = 7
patience = 50
base_path = 'models/'

# Data generator
data_generator = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
log_file_path = base_path + 'emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping(monitor='val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
trained_models_path = base_path + 'mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.keras'
model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# Loading dataset
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)

# Calculate steps per epoch
steps_per_epoch = max(1, len(xtrain) // batch_size)

# Model training
model.fit(
    data_generator.flow(xtrain, ytrain, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=(xtest, ytest)
)
