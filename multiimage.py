from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator, _PIL_INTERPOLATION_METHODS, array_to_img, img_to_array, load_img
import multiprocessing as mp
import numpy as np
import os
from keras import backend as K


class ThreadedImageDataGenerator(ImageDataGenerator):
    def __int__(self,
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-6,
                rotation_range=0.,
                width_shift_range=0.,
                height_shift_range=0.,
                brightness_range=None,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=False,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None,
                data_format=None,
                validation_split=0.0):

        super(ImageDataGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            validation_split=validation_split)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        return ThreadedDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)


class ThreadedDirectoryIterator(DirectoryIterator):
    def __init__(self,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest'):

        super(ThreadedDirectoryIterator, self).__init__(
                 directory=directory,
                 image_data_generator=image_data_generator,
                 target_size=target_size,
                 color_mode=color_mode,
                 classes=classes,
                 class_mode=class_mode,
                 batch_size=batch_size,
                 shuffle=shuffle,
                 seed=seed,
                 data_format=data_format,
                 save_to_dir=save_to_dir,
                 save_prefix=save_prefix,
                 save_format=save_format,
                 follow_links=follow_links,
                 subset=subset,
                 interpolation=interpolation)

    def _get_batches_of_transformed_samples(self, filename):
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        fname = filename
        img = load_img(os.path.join(self.directory, fname),
                       grayscale=grayscale,
                       target_size=None,
                       interpolation=self.interpolation)
        if self.image_data_generator.preprocessing_function:
            img = self.image_data_generator.preprocessing_function(img)
        if self.target_size is not None:
            width_height_tuple = (self.target_size[1], self.target_size[0])
            if img.size != width_height_tuple:
                if self.interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            self.interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[self.interpolation]
                img = img.resize(width_height_tuple, resample)
        x = img_to_array(img, data_format=self.data_format)
        x = self.image_data_generator.random_transform(x)
        x = self.image_data_generator.standardize(x)

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            img = array_to_img(x, self.data_format, scale=True)
            fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                              index=j,
                                                              hash=np.random.randint(1e7),
                                                              format=self.save_format)
            img.save(os.path.join(self.save_to_dir, fname))


    def construct_next(self):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())

        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    # __init__ will start the workers and fill the buffer to specified amount

    # 1 worker will be used to fill the work to do queue
        # Worker will wait until all workers are complete before placing next batch of images
            # Could use 2 queue's are rotate
            # or detect if workers are idle...
    # X workers will be used for image conversion

    # next will pull batch_size from the queue and process


threads = 5

def get_datasets(x, output_queues):
    pass

input_queue = mp.Queue()
output_queue = mp.Queue()

processes = [mp.Process(target=get_datasets, args=(x, output_queue)) for x in range(len(output_queue))]
for process in processes:
    process.start()

thing = output_queue.get()
