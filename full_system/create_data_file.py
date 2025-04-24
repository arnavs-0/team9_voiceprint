import os
import csv

dataset_root = "dataset"
output_csv = "all_data.csv"

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label", "auth"])
    for folder in os.listdir(dataset_root):
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if not file.endswith(".wav"):
                continue
            filepath = os.path.join(folder_path, file)
            # Default values
            label = 2  
            auth = 0   

            # Assign labels based on folder

            if folder == "augmented_user_command": 
                # Command, user 
                label = 1 
                auth = 1
            elif folder == "negative_random": 
                # Neither, not user 
                label = 2
                auth = 0
            elif folder == "negative_wrong_speaker": 
                # Wake word, not user 
                label = 0
                auth = 0 
            elif folder == "negative_wrong_text": 
                # Neither, user 
                label = 2
                auth = 1
            elif folder == "negative_wrong_text_synthetic": 
                # Neither, not user 
                label = 2
                auth = 0
            elif folder == "positive": 
                # Wake word, user 
                label = 0 
                auth = 1
            elif folder == "positive_synthetic": 
                # Wake word, not user 
                label = 0
                auth = 0
            elif folder == "user_command": 
                # Command, user 
                label = 1 
                auth = 1

            writer.writerow([filepath, label, auth])
