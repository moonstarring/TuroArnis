# file: generate_pose_rules.py

import pandas as pd
import numpy as np

def generate_pose_definitions(csv_path, output_path, std_dev_multiplier=2.0):
    """
    analyzes the pose data from a csv file and generates the POSE_LIBRARY.

    args:
        csv_path (str): path to the input csv file with angle data.
        output_path (str): path to the output .py file to save the dictionary.
        std_dev_multiplier (float): how many standard deviations to use for the range.
                                    2.0 covers approx. 95% of the data.
    """
    print(f"[info] loading data from {csv_path}...")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[error] csv file not found at {csv_path}. please run the training script first.")
        return

    pose_library = {}
    
    # get a list of all unique poses (classes) in the dataset
    poses = data['class'].unique()
    print(f"[info] found poses: {poses}")

    for pose_name in poses:
        print(f"  -> processing pose: {pose_name}")
        
        # filter the dataframe to only include data for the current pose
        pose_data = data[data['class'] == pose_name]
        
        pose_rules = {}
        
        # loop through all angle columns (all columns except 'class')
        for joint in pose_data.columns[1:]:
            # calculate the mean and standard deviation for the joint's angles
            mean_angle = pose_data[joint].mean()
            std_dev_angle = pose_data[joint].std()
            
            # define the min and max angle based on the standard deviation
            min_angle = mean_angle - (std_dev_multiplier * std_dev_angle)
            max_angle = mean_angle + (std_dev_multiplier * std_dev_angle)
            
            # store the rule for this joint
            pose_rules[joint] = [round(min_angle, 2), round(max_angle, 2)]
        
        # add the completed rules for this pose to the main library
        pose_library[pose_name] = pose_rules

    # now, write the generated dictionary to the output python file
    print(f"\n[info] writing pose library to {output_path}...")
    with open(output_path, 'w') as f:
        f.write("# file: pose_definitions.py (auto-generated)\n\n")
        f.write("POSE_LIBRARY = {\n")
        for pose_name, rules in pose_library.items():
            f.write(f"    '{pose_name}': {{\n")
            for joint, angle_range in rules.items():
                f.write(f"        '{joint}': {angle_range},\n")
            f.write("    },\n")
        f.write("}\n")

    print("[info] successfully generated pose definitions.")

if __name__ == "__main__":
    # configure the input and output file paths
    input_csv = 'arnis_poses_for_classification.csv'
    output_py_file = 'pose_definitions.py'
    
    generate_pose_definitions(input_csv, output_py_file)