:,0# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:59:26 2018

@author: 大茄茄
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


SampFreq = 256
ChannelNum = 22

def read_onset_edf(edfpath, xlsPath, savepath):
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
            
            

def read_edf(filepath, parentFile, filename, startTime, endTime):
    f = pyedflib.EdfReader(filepath)
    channel_list = channel_handle(parentFile, filename)
    data = []
    for i in channel_list:
        per_channel = f.readSignal(i)
        per_channel = per_channel[startTime*SampFreq : endTime*SampFreq]
        data.append(per_channel)
    return data



#对原始的edf文件信道进行处理，去掉多余的信道
def channel_handle(parentFile, filename):
    files = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06',
             'chb07', 'chb08', 'chb09', 'chb10', 'chb23', 'chb24']
    if parentFile in files:
        return list(range(22))
    files = ['chb11', 'chb12', 'chb13', 'chb14', 'chb17', 'chb18',
             'chb19', 'chb20', 'chb21', 'chb22']
    if parentFile in files:
        if filename == 'chb11_01':
            return list(range(22))
        else:
            return [0,1,2,3,5,6,7,8,13,14,15,16,18,19,20,21,10,11,23,24,25,26]
    elif parentFile == 'chb15':
        return [0,1,2,3,5,6,7,8,14,15,16,17,19,20,21,22,10,11,24,25,26,27]
        
    
    
    
    
if __name__ == '__main__':
    xlsPath = r'E:\Optimize_eeg\data\summary.xlsx'
    edfpath = r'G:\chbmit_analysis\handle\handled_rawEdfData'
    onset_savepath = r'E:\Optimize_eeg\data\origin_onset_data'
#    read_onset_edf(edfpath, xlsPath, onset_savepath)
    
    import band_filter
    
    path = r'E:\Optimize_eeg\data\origin_onset_data'
    savepath = r'E:\Optimize_eeg\data\band_filter_onset_data'
#    band_filter.filterX(path, savepath)
    
    import channel_select
    
    path = r'E:\Optimize_eeg\data\band_filter_onset_data'
    entropy_savepath = r'E:\Optimize_eeg\data\result'
#    channel_select.walvet_entropy_cal(path, entropy_savepath)
    
    entropy_path = r'E:\Optimize_eeg\data\result\wavelet_entropy.csv'
    sorted_index_savepath =  r'E:\Optimize_eeg\data\result'
#    channel_select.walvet_entropy_sort(entropy_path, sorted_index_savepath)
    
    index_path = r'E:\Optimize_eeg\data\result\sorted_entropy_index.csv'
#    channel_select.index_times(index_path)