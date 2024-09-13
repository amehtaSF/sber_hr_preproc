

from ecg_preproc import ECGProcessor, get_holter_end_time, get_matching_files, get_holter_start_time
from ishneholterlib import Holter

pid = "9552"
ecg_dir = "data/raw_ecg/" + pid
ecg_filepaths = get_matching_files(ecg_dir, "*.ecg")

error_idx = []

for i, ecg_filepath in enumerate(ecg_filepaths):
    # if (i+1) <= 53:
    #     continue
    
    box_width = 50
    box_separator = "-" * box_width
    timestamp_format = "%Y-%m-%d %H:%M:%S"

    # Start of the box
    print(f"|{box_separator}|")

    # Processing information
    processing_info = f"processing PID {pid} file {i+1}/{len(ecg_filepaths)}"
    print(f"|{processing_info:<{box_width}}|")
    
    try:
        # Load Holter data
        ecg_holter = Holter(ecg_filepath)
        ecg_holter.load_data()

        # Start and end time
        start_time = get_holter_start_time(ecg_holter)
        end_time = get_holter_end_time(ecg_holter)
        print(f"|{'start_time:':<12}{start_time.strftime(timestamp_format):<{box_width-12}}|")
        print(f"|{'end_time:':<12}{end_time.strftime(timestamp_format):<{box_width-12}}|")

        # ECG length
        ecg = ecg_holter.lead[0].data
        print(f"|{'ecg len:':<12}{len(ecg):<{box_width-12}}|")

        # Processing ECG
        ecg_processor = ECGProcessor(pid=pid, ecg_signal=ecg, start_time=start_time, end_time=end_time)
        ecg_processor.run("data/proc_ecg/9552")
    except Exception as e:
        print(f"|{'Error:':<12}{str(e):<{box_width-12}}|")
        error_idx.append(i)
    # End of the box
    print(f"|{box_separator}|")

print(f"{len(ecg_filepaths)} files processed with {len(error_idx)} errors: {error_idx}")