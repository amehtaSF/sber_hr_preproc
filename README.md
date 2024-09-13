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
 - match with EMA
 - Produce QRS label plots
 - should probably manually check HR daily cycles of every participant in case time was set differently for different participants


 Pipeline can produce csvs, but csvs will be very large, they lose the memory optimizations i did to keep storage down. they will also be slower. we may also lose timestamp precision.