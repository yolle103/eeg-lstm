import numpy as np
import make_image
data = np.load('./data-combine/data.npy')
label = np.load('./data-combine/label.npy')


SampFreq = 256
ChannelNum = 22
new_data = []
for item in data:
    new_sample = []
    for i in xrange(0, 6):
        time_data = []
        for channel in item:
            new_channel = channel[i*SampFreq:(i+1)*SampFreq]
            time_data.append(new_channel)
        new_sample.append(time_data)
    new_data.append(new_sample)
print np.shape(new_data)


cnn_data = []
cnn_label = []
i = 0
for item in new_data:
    cnn_data.extend(item)
    cnn_label.extend([label[i]]*6)
    i += 1
print 'start generating image'
image_data = []
count = 0
for item in cnn_data:
    image = make_image.make_single_image(item)
    image_data.append(image)
    print 'generating {} image'.format(count)
    count += 1
print np.shape(cnn_data)
print np.shape(cnn_label)
print np.shape(image_data)
np.save('cnn_data.npy', cnn_data)
np.save('cnn_label.npy', cnn_label)
np.save('image_cnn_data.npy', image_data)
