#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:13:34 2024

@author: ravikishan
"""

import openslide
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
import cv2
import matplotlib as plt
from PIL import Image

# Define some constants
PATCH_SIZE = 224  # Size of the patches to extract from the WSI
BACKGROUND_THRESHOLD = 0.3  # Threshold for removing background patches
OUTPUT_DIR = 'output_patches'  # Directory to save the extracted patches

# Load WSI
def load_wsi(filepath):
    slide = openslide.OpenSlide(filepath)
    return slide

# Perform segmentation to isolate tissue regions
def segment_tissue(slide, level=0):
    thumbnail = slide.get_thumbnail(slide.level_dimensions[level])
    thumbnail = thumbnail.convert('RGB')
    thumbnail_np = np.array(thumbnail)
    gray_thumbnail = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)
    _, binary_thumbnail = cv2.threshold(gray_thumbnail, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Print segmented image
    segmented_image = Image.fromarray(binary_thumbnail)
    segmented_image.show()
    
    return binary_thumbnail

def extract_patches(slide, binary_mask, patch_size, stride=PATCH_SIZE, background_threshold=BACKGROUND_THRESHOLD):
    w, h = slide.dimensions
    patches = []
    for x in range(0, w, stride):
        for y in range(0, h, stride):
            if x + patch_size <= w and y + patch_size <= h:
                patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                
                # Check if the patch overlaps with tissue region
                if has_tissue(binary_mask, x, y, patch_size, background_threshold):
                    patch_np = np.array(patch.convert('RGB'))
                    patches.append((normalize_patch(patch_np), x, y))
    return patches

def has_tissue(binary_mask, x, y, patch_size, threshold):
    # Convert coordinates to match the downsampled binary mask
    downsample_factor = 1  # Assuming downsampling factor of 32 as in the original code
    x_ds = x // downsample_factor
    y_ds = y // downsample_factor
    patch_size_ds = patch_size // downsample_factor

    mask_patch = binary_mask[y_ds:y_ds + patch_size_ds, x_ds:x_ds + patch_size_ds]
    
    # Debugging step: Print the mask patch
    print("Mask Patch:")
    print(mask_patch)
    
    black_pixel_ratio = np.sum(mask_patch == 0) / (mask_patch.shape[0] * mask_patch.shape[1])
    
    # Debugging step: Print the black pixel ratio
    print(f"Black Pixel Ratio: {black_pixel_ratio}")
    
    return black_pixel_ratio > threshold

# Normalize patch
def normalize_patch(patch):
    patch = patch.astype('float32') / 255.0
    mean = np.mean(patch, axis=(0, 1), keepdims=True)
    std = np.std(patch, axis=(0, 1), keepdims=True)
    normalized_patch = (patch - mean) / (std + 1e-7)
    return normalized_patch



# Save patches
def save_patches(patches, slide_name, output_dir):
    slide_output_dir = os.path.join(output_dir, slide_name)
    os.makedirs(slide_output_dir, exist_ok=True)
    
    for i, (patch, x, y) in enumerate(patches):
        patch_filename = os.path.join(slide_output_dir, f'patch_{x}_{y}.png')
        #patch_image = (patch * 255).astype(np.uint8)
        patch_image = ((patch - patch.min()) / (patch.max() - patch.min()) * 255).astype(np.uint8)
        #cv2.imwrite(patch_filename, patch_image)
        cv2.imwrite(patch_filename, cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR))

# Main function to process multiple WSIs
def process_wsi_files(filepaths, output_dir):
    for filepath in filepaths:
        slide_name = os.path.basename(filepath).split('.')[0]
        slide = load_wsi(filepath)
        binary_mask = segment_tissue(slide)
        patches = extract_patches(slide, binary_mask, PATCH_SIZE)
        save_patches(patches, slide_name, output_dir)
        

if __name__ == "__main__":
    # List of file paths to the WSIs
    wsi_filepath = [
        '/Users/ravikishan/Desktop/1_Scan1.qptiff',
        '/Users/ravikishan/Desktop/2_Scan1.qptiff'
        
    ]

    # Create the output directory if it doesn't exist
    #os.makedirs(OUTPUT_DIR, exist_ok=True)
    output='/Users/ravikishan/Desktop/29'

    # Process the WSIs and save the patches
    process_wsi_files(wsi_filepath, output)
