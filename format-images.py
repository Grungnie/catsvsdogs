__author__ = 'Matthew'

from PIL import Image
import numpy as np
import pandas as pd

DOG = 1
CAT = 0

num_images = 1
img_rows, img_cols, num_colors = 64, 64, 1

raw_image = 'input/train/cat.0.jpg'
output_image = 'prepared/train/cat.0.png'
output_format = 'PNG'

images = None

for index in range(0, 12500):
    for type in ['dog', 'cat']:

        im = Image.open('input/train/{}.{}.jpg'.format(type, index))
        width, height = im.size   # Get dimensions


        if width > height:
            modifier = (width - height)/2
            cropped_image = im.crop((modifier, 0, modifier+height, height-1))
        elif width < height:
            modifier = (height - width)/2
            cropped_image = im.crop((0, modifier, width-1, modifier+width))
        else:
            cropped_image = im

        resized_image = cropped_image.resize(size=(img_rows, img_cols))

        raw_grey_image = resized_image.convert('LA')
        grey_image = np.array(list(raw_grey_image.getdata(band=0)), float)
        grey_image.shape = (raw_grey_image.size[1], raw_grey_image.size[0])

        numpy_array = np.array(grey_image)
        numpy_reshaped = numpy_array.reshape(img_rows*img_cols*num_colors,)

        if images is None:
            images = pd.DataFrame(columns=['name', 'class', 'data'], data=[['{}.{}'.format(type,index), 0 if type=='cat' else 1, list(numpy_reshaped)]])
        else:
            images = images.append([{'name': '{}.{}'.format(type,index), 'class': 0 if type=='cat' else 1, 'data': list(numpy_reshaped)}])

# y = np.append(y, CAT)
# x = np.append(x, numpy_array)

# x = x.reshape(num_images, img_rows, img_cols, num_colors)

# print(y)
# print(x.shape)

#print(images)

#resized_image.save(output_image, output_format)

images.to_csv('dogsvscats-train-med-grey.csv' ,index=False)
#print(images)
