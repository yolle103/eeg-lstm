import numpy as np
import os

def get_label_file_path(data_file_path):
    basename = os.path.basename(data_file_path)
    dir_name = os.path.dirname(data_file_path)
    label_basename = basename.replace('data', 'label')
    return os.path.join(dir_name, label_basename)

def combine_data(data_dir):
    file_list = []
    data_combine = []
    label_combine = []
    for file in os.listdir(data_dir):
            if 'data' in file:
                file_list.append(os.path.join(data_dir, file))
    for data_file in file_list:
        label_file = get_label_file_path(data_file)
        data = np.load(data_file)
        label = np.load(label_file)
        print data_file, data.shape, label.shape
        data_combine.extend(data)
        label_combine.extend(label)
    print np.shape(data_combine), np.shape(label_combine)
    np.save('data.npy', data_combine)
    np.save('label.npy', label_combine)

def main():
    combine_data('./data')

if __name__ == '__main__':
    main()
