from helper_functions import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import backend as K
import numpy as np

img_width, img_height = 150, 150 # dimensions of our images.
validation_data_dir = 'data/validation'
batch_size = 100
val_samples = 5000

validation_batches = val_samples // batch_size

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True)

model_dir = 'conv-64-64-128-128'

model = load_model('models/{}'.format(model_dir),
                   epoch=None,
                   best_acc=True,
                   model_to_load=None)

#result = model.evaluate_generator(validation_generator, steps=validation_batches)
#result = model.predict_generator(validation_generator, steps=validation_batches)
#print(result)

fname = 'data/validation/dogs/dog.10010.jpg'
img = load_img(fname,
               grayscale=False,     # based on colour_mode - rgb generally
               target_size=(img_width, img_height))
               #interpolation='nearest')    # Defaults to this
x = img_to_array(img, data_format=K.image_data_format())    # This is set by default
x = test_datagen.random_transform(x)
x = test_datagen.standardize(x)

result = model.predict(np.array([x]))
print(result)