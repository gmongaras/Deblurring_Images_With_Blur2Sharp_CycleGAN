import numpy as np
from PIL import Image, ImageFilter
import os
import random


# Declare variables that will be used to blur and save the images
origPath = "./images/original"
origFixedPath = "./images/original_fixed"
blurPath = "./images/blurred"
maxBlur = 6
maxBlurArray = [1,2,3,4,5,6]
maxBlurWeights = [0.1333, 0.1333, 0.1334, 0.2, 0.2, 0.2]

# For all images in the "original" images directory
for i in os.listdir(origPath):
    # Get the blur amount  for this image. To make the AI more robust,
    # the AI should see many types of blurred images. Since
    # more blurry images are the priority, blur values of 4, 5, and 6
    # are weighted so that they occur slightly more than blur values of 1
    # 2, and 3.
    #blurAmt = random.randint(0, maxBlur)
    blurAmt = random.choices(maxBlurArray, maxBlurWeights, k=maxBlur)[0]


    # Get the original image
    origImg = Image.open(os.path.join(origPath, str(i)))
    """print(origImg.size)
    origImg.show()
    input()"""

    # Since blurring the image decreases the size of the image,
    # remove the outer pixels of the original image to match up with
    # the blurred image.
    origImgFixed = np.array(origImg)
    origImgFixed = origImgFixed[maxBlur:-maxBlur, maxBlur:-maxBlur]
    origImgFixed = Image.fromarray(origImgFixed)
    """print(origImgFixed.size)
    origImgFixed.show()
    input()"""
    origImgFixed.save(os.path.join(origFixedPath, str(i)), "PNG")

    # Blur the image
    blurImg = origImg.filter(ImageFilter.BoxBlur(blurAmt))
    """print(blurImg.size)
    blurImg.show()
    input()"""

    # Remove the outer border of the blurred image
    blurImgFixed = np.array(blurImg)
    blurImgFixed = blurImgFixed[maxBlur:-maxBlur, maxBlur:-maxBlur]
    blurImgFixed = Image.fromarray(blurImgFixed)
    """print(blurImgFixed.size)
    blurImgFixed.show()
    input()"""
    blurImgFixed.save(os.path.join(blurPath, str(i)), "PNG")