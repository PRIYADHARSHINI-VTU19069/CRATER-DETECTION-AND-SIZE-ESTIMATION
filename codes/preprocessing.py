import numpy as np
import os
import cv2


# Function to read the .img file and convert it to a 2D array
def read_img_file(img_path, width, height):
    with open(img_path, 'rb') as f:
        img_data = np.fromfile(f, dtype=np.uint8)

    # Reshape the data to the correct dimensions
    img = img_data.reshape((height, width))
    return img


# Function to chunk the image into 512x512 patches and save them as PNGs
def save_image_patches(img, patch_size=512, output_folder="your/custom/folder/path"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    patch_id = 0
    for y in range(0, img.shape[0], patch_size):
        for x in range(0, img.shape[1], patch_size):
            # Extract the patch
            patch = img[y:y + patch_size, x:x + patch_size]

            # Only save patches that are exactly 512x512
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patch_filename = os.path.join(output_folder, f"patch_{patch_id:04d}.png")

                # Save the patch as a PNG file
                cv2.imwrite(patch_filename, patch)
                patch_id += 1


# Example usage:
img_path = '/Users/ansilin/Documents/my_works/MOON_taiwan/dataset/IMG_dataset/ch2_ohr_ncp_20210401T2357376656_d_img_hw1/data/calibrated/20210401/ch2_ohr_ncp_20210401T2357376656_d_img_hw1.img'  # Path to your .img file
width = 12000  # Image width
height = 90148  # Image height

# Custom output folder path where you want to save the patches
output_folder = '/Users/ansilin/Documents/my_works/MOON_taiwan/dataset/patches_output/ch2_ohr_ncp_20210401T2357376656_d_img_hw1'  # Replace with your desired folder

# Step 1: Read the image from the .img file
img = read_img_file(img_path, width, height)

# Step 2: Chunk the image and save the patches in the specified folder
save_image_patches(img, output_folder=output_folder)

