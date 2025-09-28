import os

# Dataset paths
train_path = './Detector_de_Cascos-4/train/images'
valid_path = './Detector_de_Cascos-4/valid/images'
test_path = './Detector_de_Cascos-4/test/images'

# List files in each split
train_files = os.listdir(train_path)
valid_files = os.listdir(valid_path)
test_files = os.listdir(test_path)

# Count files
num_train = len(train_files)
num_valid = len(valid_files)
num_test = len(test_files)

# Print counts
print(f"Number of files in train: {num_train}")
print(f"Number of files in valid: {num_valid}")
print(f"Number of files in test: {num_test}")
print(f"Total number of files: {num_train + num_valid + num_test}")
