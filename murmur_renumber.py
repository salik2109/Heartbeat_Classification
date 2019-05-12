import os
path = 'Murmurs/'
i = 1
for filename in os.listdir(path):
    os.rename(os.path.join(path, filename), os.path.join(path,  str(i)))
    i = i +1