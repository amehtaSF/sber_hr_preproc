

from ecg_preproc import ECGProcessor, get_holter_end_time, get_matching_files, get_holter_start_time
from ishneholterlib import Holter
import time
import os
from argparse import ArgumentParser

from logger_setup import setup_logger

OUTPUT_PROC_ECG_DIR = "data/proc_ecgs"

logger = setup_logger()
parser = ArgumentParser()
parser.add_argument("--raw_dir", type=str, 
                    required=True,
                    help="Directory containing raw ECG files. Top-level dir should be named after PID.")


script_start_ts = time.time()
logger.info(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(script_start_ts))}")


ecg_dir = parser.parse_args().raw_dir
pid = os.path.basename(ecg_dir)
assert pid.isdigit(), f"Top level directory of raw_dir should be an integer PID: {ecg_dir}."
assert os.path.exists(ecg_dir), f"Directory does not exist: {ecg_dir}"
ecg_filepaths = get_matching_files(ecg_dir, "*.ecg")

error_idx = []

for i, ecg_filepath in enumerate(ecg_filepaths):
    # if (i+1) <= 53:
    #     continue
    
    # Box formatting for logging
    box_width = 50
    box_separator = "-" * box_width
    timestamp_format = "%Y-%m-%d %H:%M:%S"

    # Start of the box
    logger.info(f"|{box_separator}|")

    # Processing information
    processing_info = f"processing PID {pid} file {i+1}/{len(ecg_filepaths)}"
    logger.info(f"|{processing_info:<{box_width}}|")
    
    try:
        # Load Holter data
        ecg_holter = Holter(ecg_filepath)
        ecg_holter.load_data()

        # Start and end time
        start_time = get_holter_start_time(ecg_holter)
        end_time = get_holter_end_time(ecg_holter)
        logger.info(f"|{'start_time:':<12}{start_time.strftime(timestamp_format):<{box_width-12}}|")
        logger.info(f"|{'end_time:':<12}{end_time.strftime(timestamp_format):<{box_width-12}}|")

        # ECG length
        ecg = ecg_holter.lead[0].data
        logger.info(f"|{'ecg len:':<12}{len(ecg):<{box_width-12}}|")

        # Processing ECG
        ecg_processor = ECGProcessor(pid=pid, ecg_signal=ecg, start_time=start_time, end_time=end_time)
        ecg_processor.run(os.path.join(OUTPUT_PROC_ECG_DIR, {pid}))
        
    except Exception as e:
        logger.error(f"|{'Error:':<12}{str(e):<{box_width-12}}|")
        error_idx.append(i)
    # End of the box
    logger.info(f"|{box_separator}|")
    
    # Calculate estimated time remaining
    cur_time = time.time()
    duration = cur_time - script_start_ts
    time_per_loop = duration / (i+1)
    remaining_time = time_per_loop * (len(ecg_filepaths) - (i+1))
    hours, remainder = divmod(remaining_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Est time remaining: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
    
    

script_end_ts = time.time()
duration = script_end_ts - script_start_ts
hours, remainder = divmod(duration, 3600)
minutes, seconds = divmod(remainder, 60)



print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(script_start_ts))}")
logger.info(f"Script ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(script_end_ts))}")
logger.info(f"Total duration: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")

logger.info(f"{len(ecg_filepaths)} files processed with {len(error_idx)} errors: {error_idx}")
