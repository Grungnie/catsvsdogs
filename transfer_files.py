__author__ = 'Matthew'

from shutil import copyfile


for index in range(12500):
    for type in ['cat', 'dog']:
        src = '/srv/samba/share/Programming/dogsvscats/input/train/{}.{}.jpg'.format(type, index)
        if index < 10000:
            dst = '/srv/samba/share/Programming/dogsvscats/data/train/{}s/{}.{}.jpg'.format(type, type, index)
        else:
            dst = '/srv/samba/share/Programming/dogsvscats/data/test/{}s/{}.{}.jpg'.format(type, type, index)
        copyfile(src, dst)