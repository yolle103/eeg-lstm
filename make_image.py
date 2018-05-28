import numpy as np
import band_filter

def make_image(data, time_step):
    '''
    input: data [sample, channel, feature]
    output image data [sample, timestep, 3, channel, sampfreq]
    '''
    image_data = []
    new_data = []
    for item in data:
        fea = item.flatten()
        slice_size = len(fea)/time_step
        new_item = []
        for i in xrange(0, time_step):
            new_item.append(fea[i*slice_size: (i+1)*slice_size])
        new_data.append(new_item)
    print np.shape(new_data)
    # new data: [sample, timestep, feature]
    def get_band_filter(data, st_feq, ed_feq):
        '''
        return [channel, SampFreq]
        '''
        out_data = []
        for i in xrange(0, ChannelNum):
            channel_data = data[i*SampFreq: (i+1)*SampFreq]
            filterd_channel_data = band_filter.butter_bandpass_filter(
                    channel_data, st_feq, ed_feq, SampFreq)
            out_data.append(filterd_channel_data)
        return out_data
    count = 0
    for sample in new_data:
        image_sample = []
        print 'process {}-th image', count
        count += 1
        for t_win in sample:
            R_data = get_band_filter(t_win, 0.01, 7)
            G_data = get_band_filter(t_win, 8, 13)
            B_data = get_band_filter(t_win, 13, 30)
            image = []
            image.append(R_data)
            image.append(G_data)
            image.append(B_data)
            image_sample.append(image)
        image_data.append(image_sample)
    np.save('image_data.npy', image_data) 

