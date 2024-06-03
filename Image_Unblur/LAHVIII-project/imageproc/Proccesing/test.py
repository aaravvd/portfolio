from tkinter import Image

import numpy as np
# import pandas as pd
# import tensorflow as tf
import tarfile
import os
import glob
from PIL import Image

# path = glob.glob("lfw-funneled.tgz")
#
# for file in path:
#     t = tarfile.open(file, 'r')
#     for member in t.getmembers():
#         if ".jpg" in member.name:
#             t.extract(member, "Extracted_Images")
#
folder_dir = "C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/betterproj/imageproc/archive/clear/"
size = 25,25
for image in os.listdir(folder_dir):
    print("C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/betterproj/imageproc/archive/clear/" + image)
    im = Image.open("C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/betterproj/imageproc/archive/clear/" + image)
    im_resized = im.resize(size)
    im_resized.save("C:/Users/kaide/OneDrive/Desktop/vulcan type shit/vulcanfinal/betterproj/imageproc/archive/lowres/" + image, "PNG")



