import os

def rename_image_files(folder_path):
    """
    Adds the prefix "1(2)_" to all image files in a specified folder.

    Args:
        folder_path (str): The absolute path to the folder containing the images.
    """
    # A list of common image file extensions to look for.
    # You can add or remove extensions as needed.
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

    try:
        # Get the list of all files in the directory.
        files = os.listdir(folder_path)

        for filename in files:
            # Check if the file is an image based on its extension.
            if filename.lower().endswith(image_extensions):
                # Create the full original path to the file.
                original_file_path = os.path.join(folder_path, filename)

                # Create the new filename with the prefix.
                new_filename = f"4(2)_{filename}"

                # Create the full new path for the renamed file.
                new_file_path = os.path.join(folder_path, new_filename)

                # Rename the file.
                os.rename(original_file_path, new_file_path)
                print(f'Renamed: "{filename}" to "{new_filename}"')

        print("\nRenaming process completed successfully.")

    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' was not found. Please double-check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- HOW TO USE ---
if __name__ == "__main__":
    # CORRECTED FOLDER PATH:
    path_to_your_folder = r"C:\Users\vsamb\Documents\TuroArnis\no classes\4 (2)"
    
    rename_image_files(path_to_your_folder)