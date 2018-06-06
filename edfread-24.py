# -*- coding: utf-8 -*-

#对原始的发作时期的患者的.edf格式文件进行读取，转换成.csv格式文件
#并对一些多余的信道进行处理

from __future__ import division
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import xlrd
import pyedflib
import argparse
import datetime
from datetime import timedelta
import time
import math
import random
import edfread
SampFreq = 256
ChannelNum = 22

def get_parser():
    parser = argparse.ArgumentParser(description='Process edf file save to npy')
    parser.add_argument('-f', '--folder', help='edf file folder')
    parser.add_argument('-s', '--save_dir', help='save dir')
    return parser.parse_args()

def read_raw_data(edf_dir):
    edf_file_list = []
    summary_file_path = ''
    # get all edf file path in folder
    dir_name = os.path.basename(edf_dir)
    for file in os.listdir(edf_dir):
            if file.endswith(".edf"):
                edf_file_list.append(os.path.join(edf_dir, file))
            if 'summary' in file:
                summary_file_path = os.path.join(edf_dir, file)
     
    print('edf file num: {}'.format(len(edf_file_list)))
    print('summary path: {}'.format(summary_file_path))
    # read summary
    file_info = read_summary(summary_file_path)
    # file info is a list of dict containing each edf file's information
    # dict format: {'filename':, 'num_seizure':, 'seizure':[(st1, end1),(st2, end2)]}

    # read edf file one by one
    seizure_raw_data = []
    no_seizure_raw_data = []
    # first save seizure data
    seizure_total_time = 0
    for file_path in edf_file_list:
        basename = os.path.basename(file_path)
        info = [item for item in file_info if item["filename"] == basename]
        if info is None or len(info) == 0:
            info = {}
            info['num_seizure'] = 0
        else:
            info = info[0]
        if info['num_seizure'] > 0:
            # extrach seizure time in file
            print('reading seizure data from {}'.format(file_path))
            for each in info['seizure']:
                print('reading seizure data start {}, end {}'.format(each[0], each[1]))
                seizure_total_time += each[1] - each[0]
                edf_data = read_edf(
                        file_path, 
                        each[0], 
                        each[1]) 
                seizure_raw_data.append(edf_data)

        else:
            # extract data from 30 - 35  minutes in file
            edf_data = read_edf(file_path, 30*60, 35*60)
            print('reading no seizure data from {}'.format(file_path))
            print('edf data shape: {}'.format(np.shape(edf_data)))
            no_seizure_raw_data.append(edf_data)
            
            
    print('seizure raw data shape {}'.format(np.shape(seizure_raw_data)))
    # extract enough none-seizure data according to seizure total time
    print('total seizure time: {}'.format(seizure_total_time))
    return seizure_raw_data, no_seizure_raw_data

def read_edf(file_path, start_time=None, end_time=None):
    f = pyedflib.EdfReader(file_path)
    channel_list = channel_handle(os.path.basename(file_path))
    data = []
    for i in channel_list:
        per_channel = f.readSignal(i)
        if not start_time is None:
            per_channel = per_channel[
                    start_time*SampFreq : end_time*SampFreq]
        data.append(per_channel)
    return data



#对原始的edf文件信道进行处理，去掉多余的信道
def channel_handle(file_path):
    basename = os.path.basename(file_path)
    head = basename.split('_')[0]
    normal_file = [
                'chb01', 'chb02', 'chb03', 'chb04', 
                'chb05', 'chb06', 'chb07', 'chb08', 
                'chb09', 'chb10', 'chb23', 'chb24']
    modify_file = [
            'chb11', 'chb12', 'chb13', 'chb14', 
            'chb17', 'chb18', 'chb19', 'chb20', 
            'chb21', 'chb22']

    if head in normal_file:
        return list(range(22))
    elif head in modify_file:   
        # handle special case of chb11_01
        if basename == 'chb11_01.edf':
            return list(range(22))
        else:
            return [0,1,2,3,5,6,7,8,13,14,15,16,18,19,20,21,10,11,23,24,25,26]
    elif 'chb15' in head:
        return [0,1,2,3,5,6,7,8,14,15,16,17,19,20,21,22,10,11,24,25,26,27]



        
def read_summary(summary_path):
    # TODO find a better way to do this!
    file_info = []
    with open(summary_path, 'r') as f:
        content = f.readlines()
    i = 0
    while i < len(content):
        if 'File Name' in content[i]:
            info = {}
            info['filename'] = content[i].split(': ')[1][:-1]
            info['num_seizure'] = int(content[i+1].split(': ')[1])
            if info['num_seizure'] > 0:
                info['seizure'] = []
                j = 0
                while j < info['num_seizure']:
                    seizure_start_time = int(content[i+2+j*2].split(': ')[1][:-8])
                    seizure_end_time = int(content[i+3+j*2].split(': ')[1][:-8])
                    info['seizure'].append((seizure_start_time, seizure_end_time))
                    j += 1
                i += 4
            else:
                i += 2
            file_info.append(info)
            
        else:
            i += 1
    return file_info



def edf_read_from_folder(edf_dir, save_dir):
    edf_file_list = []
def read_raw_data(edf_dir):
    edf_file_list = []
    summary_file_path = ''
    # get all edf file path in folder
    dir_name = os.path.basename(edf_dir)
    for file in os.listdir(edf_dir):
            if file.endswith(".edf"):
                edf_file_list.append(os.path.join(edf_dir, file))
            if 'summary' in file:
                summary_file_path = os.path.join(edf_dir, file)
     
    print('edf file num: {}'.format(len(edf_file_list)))
    print('summary path: {}'.format(summary_file_path))
    # read summary
    file_info = read_summary(summary_file_path)
    # file info is a list of dict containing each edf file's information
    # dict format: {'filename':, 'num_seizure':, 'seizure':[(st1, end1),(st2, end2)]}

    # read edf file one by one
    seizure_raw_data = []
    no_seizure_raw_data = []
    # first save seizure data
    seizure_total_time = 0
    for file_path in edf_file_list:
        basename = os.path.basename(file_path)
        info = [item for item in file_info if item["filename"] == basename]
        if info is None or len(info) == 0:
            info = {}
            info['num_seizure'] = 0
        else:
            info = info[0]
        if info['num_seizure'] > 0:
            # extrach seizure time in file
            print('reading seizure data from {}'.format(file_path))
            for each in info['seizure']:
                print('reading seizure data start {}, end {}'.format(each[0], each[1]))
                seizure_total_time += each[1] - each[0]
                edf_data = read_edf(
                        file_path, 
                        each[0], 
                        each[1]) 
                seizure_raw_data.append(edf_data)

        else:
            # extract data from 30 - 35  minutes in file
            edf_data = read_edf(file_path, 30*60, 35*60)
            print('reading no seizure data from {}'.format(file_path))
            print('edf data shape: {}'.format(np.shape(edf_data)))
            no_seizure_raw_data.append(edf_data)
            
            
    print('seizure raw data shape {}'.format(np.shape(seizure_raw_data)))
    # extract enough none-seizure data according to seizure total time
    print('total seizure time: {}'.format(seizure_total_time))
    return seizure_raw_data, no_seizure_raw_data
    summary_file_path = ''
    # get all edf file path in folder
    dir_name = os.path.basename(edf_dir)
    for file in os.listdir(edf_dir):
            if file.endswith(".edf"):
                edf_file_list.append(os.path.join(edf_dir, file))
            if 'summary' in file:
                summary_file_path = os.path.join(edf_dir, file)
     
    print('edf file num: {}'.format(len(edf_file_list)))
    print('summary path: {}'.format(summary_file_path))
    # read summary
    file_info = read_summary(summary_file_path)
    # file info is a list of dict containing each edf file's information
    # dict format: {'filename':, 'num_seizure':, 'seizure':[(st1, end1),(st2, end2)]}

    # read edf file one by one
    seizure_raw_data = []
    no_seizure_raw_data = []
    # first save seizure data
    seizure_total_time = 0
    for file_path in edf_file_list:
        basename = os.path.basename(file_path)
        info = [item for item in file_info if item["filename"] == basename]
        if len(info) == 0:
            info = {}
            info['num_seizure'] = 0
        else:
            info = info[0]
        if info['num_seizure'] > 0:
            # extrach seizure time in file
            print('reading seizure data from {}'.format(file_path))
            for each in info['seizure']:
                print('reading seizure data start {}, end {}'.format(each[0], each[1]))
                seizure_total_time += each[1] - each[0]
                edf_data = read_edf(
                        file_path, 
                        each[0], 
                        each[1]) 
                seizure_raw_data.append(edf_data)

        else:
            # extract data from 30 - 31 minutes in file
            edf_data = read_edf(file_path, 30*60, 31*60)
            print('reading no seizure data from {}'.format(file_path))
            print('edf data shape: {}'.format(np.shape(edf_data)))
            no_seizure_raw_data.append(edf_data)
            
            
    print('seizure raw data shape {}'.format(np.shape(seizure_raw_data)))
    # extract enough none-seizure data according to seizure total time
    print('total seizure time: {}'.format(seizure_total_time))
    seizure_data = []
    no_seizure_data = []
    # modify seizure data, 6 sec as a sample

    np.save('seizure_raw_data.npy', seizure_raw_data)
    seizure_data = slice_data(seizure_raw_data, 6*SampFreq)
    no_seizure_data = slice_data(no_seizure_raw_data, 6*SampFreq)

    # get equal number of no seizure data
    if np.shape(seizure_data)[0] < np.shape(no_seizure_data)[0]:
        no_seizure_data = random.sample(no_seizure_data, np.shape(seizure_data)[0])
    print('seizure data shape: {}'.format(np.shape(seizure_data)))
    print('no seizure data shape: {}'.format(np.shape(no_seizure_data)))
    label = [0]*np.shape(seizure_data)[0] + [1]*np.shape(seizure_data)[0]
    # data format (sample, channel, feature) feature is 6s data with 256 SampFreq
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    data = no_seizure_data + seizure_data
    np.save(os.path.join(save_dir, '{}_data.npy'.format(dir_name)), data)
    np.save(os.path.join(save_dir, '{}_label.npy'.format(dir_name)), label)
    

def slice_data(input_data, slice_size):
    out_data = []
    def slice_regard_channel(data, start, end):
        out = []
        for channel in data:
            chunks = channel[start:end]
            out.append(chunks)
        return out

    for item in input_data:
        raw_size = np.shape(item)[1]
        slice_num = int(math.floor(raw_size/slice_size))
        for i in range(0, slice_num):
            out_data.append(slice_regard_channel(item, i*slice_size, (i+1)*slice_size))
                    
    print('out_size {}'.format(np.shape(out_data)))
    return out_data

def edf_read_save(edf_dir, save_dir, read_option, win_size):
    dir_name = os.path.basename(edf_dir)
    seizure_raw_data, no_seizure_raw_data = read_raw_data(edf_dir)
    options = {'slice':slice_data, 'slide':slide_data}
    seizure_data = options[read_option](
            seizure_raw_data, win_size)
    no_seizure_data = options[read_option](
            no_seizure_raw_data, win_size)

    print('seizure data shape: {}'.format(np.shape(seizure_data)))
    print('no seizure data shape: {}'.format(np.shape(no_seizure_data)))
    if np.shape(seizure_data)[0] < np.shape(no_seizure_data)[0]:
        no_seizure_data = random.sample(no_seizure_data, np.shape(seizure_data)[0])

    print('seizure data shape: {}'.format(np.shape(seizure_data)))
    print('no seizure data shape: {}'.format(np.shape(no_seizure_data)))
    label = [0]*np.shape(seizure_data)[0] + [1]*np.shape(seizure_data)[0]
    # data format (sample, channel, feature) feature is 6s data with 256 SampFreq
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    data = no_seizure_data + seizure_data
    np.save(os.path.join(
        save_dir, '{}_data.npy'.format(dir_name)), data)
    np.save(os.path.join(
        save_dir, '{}_label.npy'.format(dir_name)), label)



def save_fine_tune_data(edf_dir, save_dir, read_option, win_size):
    dir_name = os.path.basename(edf_dir)
    seizure_raw_data, no_seizure_raw_data = read_raw_data(edf_dir)
    options = {'slice':slice_data, 'slide':slide_data}
    no_seizure_data = options[read_option](
            no_seizure_raw_data, win_size)

    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # data format (sample, channel, feature) feature is 6s data with 256 SampFreq
    seizure_count = 0
    seizure_size = 0
    for each in seizure_raw_data:
        # slide/slice each seizure_raw_data

        seizure_data = options[read_option]([each], win_size)
        seizure_size += len(seizure_data)
        print np.shape(seizure_data)
        np.save(os.path.join(
            save_dir, '{}_seizure_data.npy'.format(seizure_count)), seizure_data)
        seizure_count += 1

    if seizure_size < np.shape(no_seizure_data)[0]:
        no_seizure_data = random.sample(no_seizure_data, seizure_size)
    print np.shape(no_seizure_data)
    np.save(os.path.join(
        save_dir, 'normal_data.npy'), no_seizure_data)

def test():
    data = np.load('seizure_raw_data.npy')
    print('input size: {}'.format(np.shape(data)))
    x_data = slice_data(data, 6*SampFreq)

def main():
    args = get_parser()
    edfpath = args.folder
    save_dir = args.save_dir
    edf_read_3s_slide_window(edfpath, save_dir)
    
    
if __name__ == '__main__':
    main()
    
