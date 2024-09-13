import os
import zipfile
import numpy as np
import pandas as pd
import re
import fnmatch
from ishneholterlib import Holter
from datetime import datetime, timedelta
import yaml
from typing import List, Dict, Literal
import neurokit2 as nk

config = yaml.safe_load(open("config.yaml"))


def get_matching_files(directory, pattern) -> List[str]:
    """
    Get all files in the directory that match the pattern.
    """
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def extract_ecg_zip(zip_file, extract_dir=None) -> bool:
    """
    Extract ECG zip file.
    """
    if extract_dir is None:
        extract_dir = os.path.dirname(zip_file)
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return True
    except Exception as e:
        print(f'Error extracting {zip_file}: {e}')
        return False

def get_holter_start_time(ecg_holter) -> datetime:
    """
    Get the start time from an ECG holter object
    """
    start_ts = datetime(ecg_holter.record_date.year,
                        ecg_holter.record_date.month,
                        ecg_holter.record_date.day,
                        ecg_holter.start_time.hour,
                        ecg_holter.start_time.minute,
                        ecg_holter.start_time.second)
    return start_ts

def get_holter_end_time(ecg_holter) -> datetime:
    """
    Get the end time from an ECG holter object
    """
    start_ts = get_holter_start_time(ecg_holter)
    end_ts = start_ts + ecg_holter.get_length()
    return end_ts

def get_holter_time_metadata(directory) -> List[Dict[str, str | datetime | timedelta]]:
    """
    Loop through all .ecg files in the directory and extract the start and end times.
    Return a list of dicts with keys:
        - dir (str)
        - filename (str)
        - start_time (datetime)
        - end_time (datetime) 
        - duration (timedelta)
    """
    times = []  
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, "*.ecg"):
            ecg_file = os.path.join(root, filename)
            ecg_holter = Holter(ecg_file)
            start_time = get_holter_start_time(ecg_holter)
            end_time = get_holter_end_time(ecg_holter)
            duration = ecg_holter.get_length()
            times.append({'dir': root, 
                          'filename': filename, 
                          'start_time': start_time, 
                          'end_time': end_time,
                          'duration': duration})
    return times

def get_time_interval_holters(directory: str, interval_start: datetime, interval_end: datetime) -> List[Dict[str, str | datetime | timedelta]]:
    """
    Get all holter files in the directory that contain data within the time interval.
    Returns a list of dicts that contain the filename and 
    the start and end times of the the subset of the interval that is within the file.
    """
    dir_metadatas = get_holter_time_metadata(directory)
    interval_files = []
    for file in dir_metadatas:
        if file['start_time'] < interval_end and file['end_time'] > interval_start:
            start_time = max(file['start_time'], interval_start)
            end_time = min(file['end_time'], interval_end)
            interval_files.append({
                'dir': file['dir'],
                'filename': file['filename'],
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            })
    return interval_files

def get_time_interval_files(directory: str, pattern: str, interval_start: datetime, interval_end: datetime) -> List[Dict[str, str | datetime | timedelta]]:
    """
        
    """
    fpaths = get_matching_files(directory=directory, pattern=pattern)
    interval_files = []
    for fpath in fpaths:
        f = os.path.basename(fpath)
        try:
            start_time = datetime.strptime(f.split('_')[3] + f.split('_')[4], '%Y%m%d%H%M%S')
            end_time = datetime.strptime(f.split('_')[5] + f.split('_')[6], '%Y%m%d%H%M%S')
        except ValueError:
            print(f"Error parsing start and end times for {f}")
            continue
        if start_time < interval_end and end_time > interval_start:
            start_time = max(start_time, interval_start)
            end_time = min(end_time, interval_end)
            interval_files.append({
                'dir': directory,
                'filename': f,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            })
    




class ECGProcessor:
    

    def __init__(self, ecg_signal: np.array, pid: str, start_time: datetime, end_time: datetime):
        
        self.ecg_raw = ecg_signal
        self.ecg_clean = None
        self.powerline_freq = config['powerline_hz']
        self.fs = config['ecg_hz']
        self.zhao_dict = {'Unacceptable': 0, 'Barely acceptable': 1, 'Excellent': 2}
        self.pid = pid
        self.start_time = start_time
        self.end_time = end_time
        
        # Error checking
        if ecg_signal is None or len(ecg_signal) == 0:
            raise ValueError(f"ECG signal is empty for {pid} from {start_time} to {end_time}")
        
        if len(ecg_signal) != (end_time - start_time).seconds * self.fs:
            raise ValueError(f"ECG signal length ({len(ecg_signal)}) does not match time interval for {pid} from {start_time} to {end_time}")
        
    
    def clean(self, ecg):
        '''
        Using neurokit method:
        - .5 Hz highpass filter
        - 60 Hz notch filter for powerline noise
        '''
        self.ecg_clean = nk.ecg_clean(ecg_signal=ecg, 
                                      sampling_rate=self.fs, 
                                      method='neurokit',
                                      powerline=self.powerline_freq)
        return self.ecg_clean
    
    def make_filename(self, pid: str, extension: Literal['csv', 'parquet'],
                      datatype: Literal['rpeaks', 'qrs', 'phase', 'meta'], 
                      start_time: datetime, end_time: datetime, 
                      pr_excellent: float, pr_acceptable: float, pr_unacceptable: float):
        
        start_time = start_time.strftime('%Y%m%d_%H%M%S')
        end_time = end_time.strftime('%Y%m%d_%H%M%S')
        filename = f"ECG_{pid}_{datatype}_{start_time}_{end_time}"
        if datatype == 'meta':
            filename += f"_{pr_excellent}_{pr_acceptable}_{pr_unacceptable}"
        filename += f".{extension}"
        return filename
        
    def make_file(self, filetype: Literal['csv', 'parquet'], df: pd.DataFrame, datatype: Literal['rpeaks', 'qrs', 'phase', 'meta'], directory: str=None):
        
        if datatype == 'meta':
            pr_excellent = np.round(np.sum(df['quality']==self.zhao_dict['Excellent'])/len(df), 2)
            pr_excellent = int(pr_excellent*100)
            pr_acceptable = np.round(np.sum(df['quality']==self.zhao_dict['Barely acceptable'])/len(df), 2)*100
            pr_acceptable = int(pr_acceptable*100)
            pr_unacceptable = np.round(np.sum(df['quality']==self.zhao_dict['Unacceptable'])/len(df), 2)*100
            pr_unacceptable = int(pr_unacceptable*100)
        else:
            pr_excellent = pr_acceptable = pr_unacceptable = None
        
        filename = self.make_filename(pid=self.pid,
                                      extension=filetype,
                                      datatype=datatype, 
                                      start_time=self.start_time, 
                                      end_time=self.end_time, 
                                      pr_excellent=pr_excellent, 
                                      pr_acceptable=pr_acceptable,
                                      pr_unacceptable=pr_unacceptable)
        filepath = os.path.join(directory, filename) if directory else filename
        
        if filetype == 'parquet':
            df.to_parquet(filepath)
        elif filetype == 'csv':
            df.to_csv(filepath, index=False)
        else:
            raise NotImplementedError(f"Filetype {filetype} not implemented.")
        
    
    def run(self, output_dir: str):
        """
        Produces files:
        - ECG signal
        - Quality ratings
        - R-peaks
        - QRS complexes
        - Cardiac phase annotations
        
        Filename format: ECG_{filetype}_{pid}_{start_time}_{end_time}_{pr_excellent}_{pr_acceptable}_{pr_unacceptable}.csv
        """
        
        # Clean signal (highpass and notch filter)
        ecg = self.clean(self.ecg_raw)
        ecg = ecg.astype(np.float32)
        
        # Epoch start indices
        epoch_idx = np.arange(0, len(ecg), self.fs*config['epoch_seconds'], dtype=np.int32)
        epoch_arr = np.repeat(np.arange(len(epoch_idx), dtype=np.int16), self.fs*config['epoch_seconds'])[:len(ecg)]
        
        # Detect R-peaks
        # df_rpeaks, rpeaks = nk.ecg_peaks(ecg, sampling_rate=self.fs, method='emrich2023') # originally wanted this, but got an error on one file randomly
        df_rpeaks, rpeaks = nk.ecg_peaks(ecg, sampling_rate=self.fs, method='rodrigues2021')
        df_rpeaks['ECG_R_Peaks'] = df_rpeaks['ECG_R_Peaks'].astype(np.int8)
        rpeaks = rpeaks['ECG_R_Peaks']
        
        # Get Zhao 2018 quality ratings
        quality_arr = np.full(len(ecg), -1, dtype=np.int8)
        for start_idx in epoch_idx:
            end_idx = start_idx+self.fs*config['epoch_seconds']
            epoch = ecg[start_idx:end_idx]
            quality = nk.ecg_quality(ecg_cleaned=epoch, rpeaks=rpeaks[start_idx:end_idx], sampling_rate=self.fs, method='zhao2018')
            quality_arr[start_idx:end_idx] = self.zhao_dict[quality]
            
        # Detect QRS complexes
        df_qrs, qrs = nk.ecg_delineate(ecg, rpeaks=rpeaks, 
                               sampling_rate=self.fs, 
                               method='dwt') # discrete wavelet transform (also available: continuous wavelet transform, peak based method)
        
        # Detect cardiac phase
        # https://neuropsychology.github.io/NeuroKit/functions/ecg.html#neurokit2.ecg.ecg_phase
        df_phase = nk.ecg_phase(ecg, rpeaks=rpeaks, delineate_info=qrs, sampling_rate=self.fs)
        
        # Make signal dataframe
        df_signal = pd.DataFrame({'ecg': ecg})  # 32 bit precision
        
        # Make quality and timestamp dataframe
        df_meta = pd.DataFrame(
            {'timestamp': pd.to_datetime(np.arange(len(ecg)) / self.fs, unit='s', origin=self.start_time),  # datetime format preserves nanosecond precision
             'quality': quality_arr,  # 8 bit precision
             'epoch': epoch_arr  # 16 bit precision
             })
        
        # Save files
        self.make_file(filetype='parquet', df=df_meta, datatype='meta', directory=output_dir)
        self.make_file(filetype='parquet', df=df_signal, datatype='signal', directory=output_dir)
        self.make_file(filetype='parquet', df=df_rpeaks, datatype='rpeaks', directory=output_dir)
        self.make_file(filetype='parquet', df=df_qrs, datatype='qrs', directory=output_dir)
        self.make_file(filetype='parquet', df=df_phase, datatype='phase', directory=output_dir)
        
        