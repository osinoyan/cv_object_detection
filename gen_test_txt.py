from os import listdir
from os.path import isfile, join
mypath = 'data/test'
onlyfiles = ['data/test/' +
             f for f in sorted(listdir(mypath)) if isfile(join(mypath, f))]
print(onlyfiles[-1])

darknet_data_path = '/home/apple/lee/cvHw2/darknet/data/'
with open(darknet_data_path + 'test.txt', 'w') as ff:
    for i in range(13068):
        ff.write('data/test/' + str(i+1) + '.png\n')
