import os

# The name of the main dataset directory
base_folder = 'dataset_multiclass'

# 1. The list of base pose names you provided
pose_names = [
    "left temple block",
    "right temple block",
    "left elbow block",
    "right elbow block",
    "solar plexus thrust",
    "left chest thrust",
    "right chest thrust",
    "left knee block",
    "right knee block",
    "left eye thrust",
    "right eye thrust",
    "crown thrust"
]

# Create the base folder first if it doesn't exist
if not os.path.exists(base_folder):
    os.mkdir(base_folder)
    print(f"Created base directory: {base_folder}")

# 2. Loop through each pose name in your list
for pose in pose_names:
    # Format the name: convert to lowercase and replace spaces with underscores
    # e.g., "Left Temple Block" -> "left_temple_block"
    formatted_name = pose.lower().replace(" ", "_")
    
    # 3. Create the two variations: '_correct' and '_incorrect'
    correct_folder_name = f"{formatted_name}_correct"
    incorrect_folder_name = f"{formatted_name}_incorrect"
    
    # Create the full paths for both folders
    correct_path = os.path.join(base_folder, correct_folder_name)
    incorrect_path = os.path.join(base_folder, incorrect_folder_name)
    
    # 4. Create the 'correct' folder if it doesn't exist
    if not os.path.exists(correct_path):
        os.mkdir(correct_path)
        print(f"Created folder: {correct_path}")
    else:
        print(f"Folder already exists: {correct_path}")
        
    # 5. Create the 'incorrect' folder if it doesn't exist
    if not os.path.exists(incorrect_path):
        os.mkdir(incorrect_path)
        print(f"Created folder: {incorrect_path}")
    else:
        print(f"Folder already exists: {incorrect_path}")

print("\nFolder setup complete. You now have 24 pose folders ready for your images.")