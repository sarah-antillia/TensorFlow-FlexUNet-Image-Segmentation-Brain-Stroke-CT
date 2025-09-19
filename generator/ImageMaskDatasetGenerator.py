# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2025/09/19
# ImageMaskDatasetGenerator:


import os
import io
import sys
import glob
import numpy as np
import cv2
from PIL import Image, ImageOps

import shutil
import traceback

class ImageMaskDatasetGenerator:

  def __init__(self, overlay_dir, 
                png_dir,
                output_images_dir, 
                output_masks_dir,
                category = ""):
     
     self.mask_format = "bgr"
       
     self.overlay_dir = overlay_dir
     self.png_dir     = png_dir
     self.category    = category

     self.seed      = 137
     self.W         = 512
     self.H         = 512
     #BGR color_map                  red                green
     self.color_map = {"Bleeding":(0,0,255), "Ischemia":(0,255,0)}

     self.output_images_dir = output_images_dir
     self.output_masks_dir  = output_masks_dir


  def generate(self):
    print("=== generate ")
    self.image_index = 10000
    self.mask_index  = 10000

    overlay_files = glob.glob(self.overlay_dir + "/*.png")
    overlay_files = sorted(overlay_files)
     
    png_files     = glob.glob(self.png_dir + "/*.png")
    png_files     = sorted(png_files)
    num_overlay_files= len(overlay_files)
    num_png_files    = len(png_files)
    print("=== num_image_files {}".format(num_overlay_files))
    print("=== num_mask_files   {}".format(num_png_files))
      
    if num_overlay_files != num_png_files:
       error = "Unmatched image_files and mask_files"
       print(">>> Error {}".format(error))
       raise Exception(error)

    color   = self.color_map[self.category]

    for i in range(num_overlay_files):
        overlay_file = overlay_files[i] 
        basename = os.path.basename(overlay_file)

        filename = self.category + "_" + basename
        # Modifed the following line to get a proper mask file name corresponding to the image_file.
        png_file = png_files[i]

        overlay = cv2.imread(overlay_file)
        overlay = cv2.resize(overlay, (self.W, self.H))
        
        png     = cv2.imread(png_file)
        png     = cv2.resize(png, (self.W, self.H))

        image_filepath = os.path.join(self.output_images_dir,  filename)
        cv2.imwrite(image_filepath, png) 
        print("=== Saved {}".format(image_filepath))

        # Subtract a plain png image from an overlay image to generate a mask
        mask    = overlay - png
        mask    = self.create_categorized_bgr_mask(mask, color)
        mask_filepath = os.path.join(self.output_masks_dir, filename)
        cv2.imwrite(mask_filepath , mask)
        print("=== Saved {}".format(mask_filepath))
 
  def create_categorized_bgr_mask(self, mask, color):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
    mask     = self.colorize_mask(mask, color=color, gray=255)

    return mask
  
  def colorize_mask(self, mask, color=(255, 255, 255), gray=0):
    h, w = mask.shape[:2]
    rgb_mask = np.zeros((w, h, 3), np.uint8)
    condition = (mask[...] >= gray-10) & (mask[...] <= gray+10)   
    rgb_mask[condition] = [color]  
    return rgb_mask   
  

if __name__ == "__main__":
  try:
    categories       = ["Bleeding", "Ischemia"]

    overlay_base_dir = "./Brain_Stroke_CT_Dataset/"
    images__base_dir = "./Brain_Stroke_CT_Dataset/"

    output_dir       = "./Brain-Stroke-CT-master/"
    images_dir       = "./Brain-Stroke-CT-master/images"
    masks_dir        = "./Brain-Stroke-CT-master/masks"

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  = os.path.join(output_dir, "masks")

    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)
  
    for category in categories:
      overlay_dir = overlay_base_dir + "/" + category + "/OVERLAY/"
      png_dir     = overlay_base_dir + "/" + category + "/PNG/"
      generator = ImageMaskDatasetGenerator(overlay_dir, 
                                                 png_dir,
                                                 output_images_dir,
                                                 output_masks_dir,
                                                 category = category)
      generator.generate()

  except:
    traceback.print_exc()
