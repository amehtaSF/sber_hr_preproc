
import os
from ecg_preproc import get_matching_files, extract_ecg_zip


# extract directories which are pids (i.e., integers)
pids = [f for f in os.listdir("data/raw_zip") if f.isdigit()]
print(f"Found {len(pids)} pids.")
print(f"PIDs found: {pids}")

# init counters
zip_counter = 0
error_counter = 0

# loop through pids
for pid in pids:
    zip_dir = os.path.join("data/raw_zip", pid)
    zip_filepaths = get_matching_files(zip_dir, "*ECG_ISHNE.zip")
    
    # loop through zip files and extract
    for zip_filepath in zip_filepaths:
        ecg_dir = os.path.join("data/raw_ecg", pid)
        try:
            extract_ecg_zip(zip_filepath, extract_dir=ecg_dir)
            print(f"Extracted {zip_filepath} to {ecg_dir}")
            zip_counter += 1
        except Exception as e:
            print(f"Error extracting {zip_filepath}: {e}")
            error_counter += 1

print(f"Extracted {zip_counter} zip files with {error_counter} errors.")
