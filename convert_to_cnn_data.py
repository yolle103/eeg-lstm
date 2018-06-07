import numpy as np
import os
import make_image
import combine_data

ChannelNum = 22
SampFreq = 256

def load_data(x, y):
    data = np.load(x)
    label = np.load(y)
    return data, label

def generate_train_val(train_x, train_y, val_x, val_y, out_dir):
    t_data, t_label = load_data(train_x, train_y)
    v_data, v_label = load_data(val_x, val_y)
    image_t_data, image_t_label = generate_image_data(t_data, t_label)
    image_v_data, image_v_label = generate_image_data(v_data, v_label)
    basename = os.path.basename(train_x)[:5]
    dir_name = os.path.join(out_dir, basename)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    np.save(
            os.path.join(dir_name, 'train_data.npy'), image_t_data)
    np.save(
            os.path.join(dir_name, 'train_label.npy'), image_t_label)
    np.save(
            os.path.join(dir_name, 'val_data.npy'), image_v_data)
    np.save(
            os.path.join(dir_name, 'val_data.npy'), image_v_label)

def generate_image_data(data, label):
    new_data = []
    for item in data:
        new_sample = []
        for i in range(0, 6):
            time_data = []
            for channel in item:
                new_channel = channel[i*SampFreq:(i+1)*SampFreq]
                time_data.append(new_channel)
            new_sample.append(time_data)
        new_data.append(new_sample)
    print(np.shape(new_data))
    i = 0
    cnn_data = []
    cnn_label = []
    for item in new_data:
        cnn_data.extend(item)
        cnn_label.extend([label[i]]*6)
        i += 1
    image_data = []
    count = 0
    for item in cnn_data:
        image = make_image.make_single_image(item)
        image_data.append(image)
        print('generating {} image'.format(count))
        count += 1
    return image_data, cnn_label
def gen_image(data):
    image_data = []
    count = 0
    for item in data:
        image = make_image.make_single_image(item)
        image_data.append(image)
        print('generating {} image'.format(count))
        count += 1
    return image_data

def main():
    origindata_dir = './fine_tune_win'
    out_dir = './fine_tune_win_image'

    for dirpath, dirnames, filenames in os.walk(origindata_dir):
        structure = os.path.join(out_dir, dirpath[len(origindata_dir):][1:])
        print(structure)
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder does already exits!")
    folder_list = list(range(1, 25))
    folder_list.remove(16)
    print(folder_list)
    for folder in folder_list:
        out_dir_name = os.path.join(out_dir, 'chb{0:02}'.format(folder))
        in_dir_name = os.path.join(origindata_dir, 'chb{0:02}'.format(folder))
        for file in os.listdir(in_dir_name):
            if '.npy' in file:
                data = np.load(os.path.join(in_dir_name, file))
                image_data = gen_image(data)
                np.save(os.path.join(out_dir_name, file), image_data)
#
#    file_list = []
#    image_dict = {}
#    for file in os.listdir(origindata_dir):
#            if 'data' in file:
#                file_list.append(os.path.join(origindata_dir, file))
#
#    for file in file_list:
#        basename = os.path.basename(file)[:5]
#        print('generating cnn set for {}'.format(basename))
#        data_path = file
#        label_path = combine_data.get_label_file_path(data_path)
#        o_data, o_label = load_data(data_path, label_path)
#        data, label = generate_image_data(o_data, o_label)
#
#        np.save(os.path.join(
#            out_dir, '{}_data.npy'.format(basename)), data)
#
#        np.save(os.path.join(
#            out_dir, '{}_label.npy'.format(basename)), label)




if __name__ == '__main__':
    main()
