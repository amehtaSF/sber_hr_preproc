Pipeline instructions
===

We will create a series of files so that a file contains no more than 1 hour. We will have a function which can take any start and end timestamp and construct a new dataset on the fly.

 - Find data
 - Unzip
 - Read in data
 - Process ECG signal
    - clean - ecg_clean()
    - peak detection - ecg_peaks()
    - heart rate calculation - signal_rate()
    - signal quality assessment - ecg_quality()
    - QRS complex delineation - ecg_delineate()
    - cardiac phase determination - ecg_phase()
 - Assign unix timestamp



 TODO
 - Produce QRS label plots
 - should probably manually check HR daily cycles of every participant in case time was set differently for different participants


Pipeline
===

- IT copies everything from /SPL-SBER_Online/Data Analysis/Cardiac Scout - Metadata/* to /oak/stanford/groups/gross/sber/sber_online/data/ecg/raw/
- We write python or bash to organize this data. 
    - delete PIDs that are not in our valid PID list (i.e. in final EMA data)
    - unzip all files and delete the zipped versions
    - there might be some people who have already unzipped .ecg files in their directory. we will make sure that after unzipping, we didn't create duplicate .ecg files. if so, we delete duplicates.
- Next, we will scan all files and create a csv that has the duration of recording for each file and thus we can calculate total duration for each participant. This also gives us a chance to detect errors reading the files before we begin the full preproc.
- Run preprocessing on each ecg file. for each participant, generate a report with:
    - duration of recording, proportion of each quality rating, aggregate all errors or warnings that arose.
    - inspect any errors that arose
    - sample ECG recordings with QRS labeled
    - plot HR over time of the whole 2 weeks
    - plot average HR per hour of the day to confirm time zone is correct
- Make utilities for users
    - script to generate time intervals of a given size for a list of timestamps (ie ema timestamps)
    - script to generate list of files, idx, timestamps for a list of time intervals
    - script to generate ECG plot PDFs for a given interval for manual inspection
    - script to generate scp commands for the files they need given a list of time intervals
    - Documentation of these scripts for users