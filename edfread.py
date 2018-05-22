# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:59:26 2018

@author: 大茄茄

Edit  
"""

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

def get_parser():
    parser = argparse.ArgumentParser(description='Process edf file save to npy')
    parser.add_argument('-f', '--folder', help='edf file folder')
    parser.add_argument('-s', '--save_dir', help='save dir')
    return parser.parse_args()



SampFreq = 256
ChannelNum = 22

def read_onset_edf(edf_dir, summary_path, save_dir):
    book = xlrd.open_workbook(xlsPath)
    sh = book.sheet_by_index(0)
    rows = sh.nrows
    count = 1
    while count<rows:
        onset_times = sh.cell_value(rowx=count, colx=6)
        if onset_times:
            onset_times = int(onset_times)
            for i in range(onset_times):
                filename = sh.cell_value(rowx=count, colx=3)
                parentFile = filename.split('_')[0]
                filepath = os.path.join(edfpath, parentFile, filename) + '.edf'
                startTime = int(sh.cell_value(rowx=count, colx=4))
                endTime = int(sh.cell_value(rowx=count, colx=5))
                print('{} : {} '.format(filename, str(endTime-startTime)))
                edfFile = read_edf(filepath, parentFile, filename, startTime, endTime) 
                save_to_numpy(edfFile, filepath, './data')
                onsetData = pd.DataFrame(edfFile)
                onsetData.to_csv(savepath + '\\{}_{}_onset.csv'.format(filename, str(i+1)))
                count += 1
        else:
            filename = sh.cell_value(rowx=count, colx=3)
            parentFile = filename.split('_')[0]
            filepath = os.path.join(edfpath, parentFile, filename) + '.edf'
            startTime = int(sh.cell_value(rowx=count, colx=4))
            endTime = int(sh.cell_value(rowx=count, colx=5))
            print('{} : {} '.format(filename, str(endTime-startTime)))
            edfFile = read_edf(filepath, parentFile, filename, startTime, endTime) 
            onsetData = pd.DataFrame(edfFile)
            onsetData.to_csv(savepath + '\\{}_onset.csv'.format(filename))
            count += 1  
            
def save_to_numpy(edf_file_data, edf_file_name, save_dir):
   pass 

def read_edf(file_path, start_time, end_time):
    f = pyedflib.EdfReader(file_path)
    channel_list = channel_handle(os.path.basename(file_path))
    data = []
    for i in channel_list:
        per_channel = f.readSignal(i)
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
        return list(xrange(22))
    elif head in modify_file:   
        # handle special case of chb11_01
        if file_path == 'chb11_01':
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
            info['num_seizure'] = int(content[i+3].split(': ')[1])
            print info
            if info['num_seizure'] > 0:
                seizure_start_time = int(content[i+4].split(': ')[1][:-9])
                seizure_end_time = int(content[i+5].split(': ')[1][:-9])
                info['seizure'] = []
                info['seizure'].append((seizure_start_time, seizure_end_time))
                i += 6
            else:
                i += 4
            file_info.append(info)
            
        else:
            i += 1
    return file_info



def edf_read_from_folder(edf_dir, save_dir):
    edf_file_list = []
    summary_file_path = ''
    # get all edf file path in folder
    for file in os.listdir(edf_dir):
            if file.endswith(".edf"):
                edf_file_list.append(os.path.join(edf_dir, file))
            if 'summary' in file:
                summary_file_path = os.path.join(edf_dir, file)
     
    print('edf file num: {}'.format(len(edf_file_list)))
    print('summary path: {}'.format(summary_file_path))
    # read summary
    file_info = read_summary(summary_file_path)

    # read edf file one by one
    seizure_data = []
    no_seizure_data = []
    for file_path in edf_file_list:
        basename = os.path.basename(file_path)
        info = (item for item in file_info if item["filename"] == basename).next()
        if info['num_seizure'] == 0:
            print 'no seizure', file_path
        else:
            for each in info['seizure']:
                edf_data = read_edf(
                        file_path, 
                        each[0], 
                        each[1]) 
                

def main():
    args = get_parser()
    edfpath = args.folder
    save_dir = args.save_dir
    edf_read_from_folder(edfpath, save_dir)
    
    
    
if __name__ == '__main__':
    main()
    
