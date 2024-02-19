import os
import mne
import numpy as np

# Folder path containing EEG .set files
folder_path = '/home/lasii/Research/dataset/alz/A'

# List all .set files in the folder and sort them
set_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.set')], key=lambda x: int(x.split('.')[0]))

# Initialize an empty list to store EEG data
eeg_data_list = []
eeg_data_label = []
time = 30
overlap_factor = 0.75 # 25%



def slaughter(signal, p_id):
    fs = 500
    chunk_duration = time*fs # in samples
    overlap = int(chunk_duration * overlap_factor)
    temp = []
    temp_label = []

    for j in range(0, signal.shape[1], overlap):
        if j<signal.shape[1]-chunk_duration:
            temp.append(signal[:,j:j+chunk_duration])
            temp_label.append(p_id)
                
    return np.array(temp), np.array(temp_label)
        
        


# Loop through each sorted .set file and load EEG data
for set_file in set_files:
    eeg_file_path = os.path.join(folder_path, set_file)
    raw = mne.io.read_raw_eeglab(eeg_file_path, preload=True).get_data()
    chunks, p_id = slaughter(raw, set_files.index(set_file))
    eeg_data_list.extend(chunks)
    eeg_data_label.extend(p_id)


# Convert the list of EEG data arrays to a single numpy array
combined_eeg_data = np.array(eeg_data_list)
combined_eeg_label = np.array(eeg_data_label)


# Print the shape of the combined numpy array (channels x time points)
print(combined_eeg_data.shape)
print(combined_eeg_label.shape)


np.save('/home/lasii/Research/dataset/alz/processed/A.npy' , combined_eeg_data, fix_imports=True)
np.save('/home/lasii/Research/dataset/alz/processed/A_l.npy' , combined_eeg_label, fix_imports=True)



##########################################################################################################################################



# Folder path containing EEG .set files
folder_path = '/home/lasii/Research/dataset/alz/C'

# List all .set files in the folder and sort them
set_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.set')], key=lambda x: int(x.split('.')[0]))

# Initialize an empty list to store EEG data
eeg_data_list = []
eeg_data_label = []

def slaughter(signal, p_id):
    fs = 500
    chunk_duration = time*fs # in samples
    overlap = int(chunk_duration * overlap_factor)
    temp = []
    temp_label = []

    for j in range(0, signal.shape[1], overlap):
        if j<signal.shape[1]-chunk_duration:
            temp.append(signal[:,j:j+chunk_duration])
            temp_label.append(p_id)
                
    return np.array(temp), np.array(temp_label)
        
        
        


# Loop through each sorted .set file and load EEG data
for set_file in set_files:
    eeg_file_path = os.path.join(folder_path, set_file)
    raw = mne.io.read_raw_eeglab(eeg_file_path, preload=True).get_data()
    chunks, p_id = slaughter(raw, set_files.index(set_file))
    eeg_data_list.extend(chunks)
    eeg_data_label.extend(p_id)


# Convert the list of EEG data arrays to a single numpy array
combined_eeg_data = np.array(eeg_data_list)
combined_eeg_label = np.array(eeg_data_label)


# Print the shape of the combined numpy array (channels x time points)
print(combined_eeg_data.shape)
print(combined_eeg_label.shape)

np.save('/home/lasii/Research/dataset/alz/processed/C.npy' , combined_eeg_data, fix_imports=True)
np.save('/home/lasii/Research/dataset/alz/processed/C_l.npy' , combined_eeg_label, fix_imports=True)

############################################################################################################################################





# Folder path containing EEG .set files
folder_path = '/home/lasii/Research/dataset/alz/F'

# List all .set files in the folder and sort them
set_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.set')], key=lambda x: int(x.split('.')[0]))

# Initialize an empty list to store EEG data
eeg_data_list = []
eeg_data_label = []

def slaughter(signal, p_id):
    fs = 500
    chunk_duration = time*fs # in samples
    overlap = int(chunk_duration * overlap_factor)
    temp = []
    temp_label = []

    for j in range(0, signal.shape[1], overlap):
        if j<signal.shape[1]-chunk_duration:
            temp.append(signal[:,j:j+chunk_duration])
            temp_label.append(p_id)
                
    return np.array(temp), np.array(temp_label)
        
        
        
        


# Loop through each sorted .set file and load EEG data
for set_file in set_files:
    eeg_file_path = os.path.join(folder_path, set_file)
    raw = mne.io.read_raw_eeglab(eeg_file_path, preload=True).get_data()
    chunks, p_id = slaughter(raw, set_files.index(set_file))
    eeg_data_list.extend(chunks)
    eeg_data_label.extend(p_id)


# Convert the list of EEG data arrays to a single numpy array
combined_eeg_data = np.array(eeg_data_list)
combined_eeg_label = np.array(eeg_data_label)


# Print the shape of the combined numpy array (channels x time points)
print(combined_eeg_data.shape)
print(combined_eeg_label.shape)

np.save('/home/lasii/Research/dataset/alz/processed/F.npy' , combined_eeg_data, fix_imports=True)
np.save('/home/lasii/Research/dataset/alz/processed/F_l.npy' , combined_eeg_label, fix_imports=True)


############################################################################################################################################

import numpy as np

A = np.load('/home/lasii/Research/dataset/alz/processed/A.npy')
C = np.load('/home/lasii/Research/dataset/alz/processed/C.npy')
F = np.load('/home/lasii/Research/dataset/alz/processed/F.npy')

data = np.concatenate((A,C,F))
label = np.concatenate((np.zeros(A.shape[0]), np.ones(C.shape[0]), np.ones(F.shape[0])*2))


np.save('/home/lasii/Research/dataset/alz/processed/data.npy' , data, fix_imports=True)
np.save('/home/lasii/Research/dataset/alz/processed/label.npy' , label, fix_imports=True)








