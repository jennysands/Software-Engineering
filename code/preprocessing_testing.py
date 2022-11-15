'''Preprocessing for Testing on GPU'''
from PIL import Image
import cv2
import pandas as pd
from skimage import data
from skimage.color import rgb2gray
from skimage import io
import skimage
import os, sys
import glob
import matplotlib.image as mpimg
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import expand_dims

print("Done importing")


'''Create variables'''
data_test = []
labels_test = []


df = pd.read_csv(r'/home/u668954/Merged_testing.csv')
p=0
for path in df["Path"]:
  print(p)
  img = mpimg.imread(path)

  '''Make the image gray, to reduce computation complexity'''
  gray_image, gray_ID = skimage.color.rgb2gray(img), df["ID"].iloc[p]
  gray_image = np.array(gray_image)
  data_test = np.append(data_test, gray_image)
  labels_test.append(gray_ID)


  '''Normalize the image'''
  norm_image, norm_ID = (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image)), df["ID"].iloc[p]
  norm_image = np.array(norm_image)
  data_test = np.append(data_test, norm_image)
  labels_test.append(norm_ID)


  '''Add noise to image'''
  img2 = cv2.imread(path,0)

  img_resize = cv2.resize(img2, (224, 224))

  gauss_noise=np.zeros((224,224),dtype=np.uint8)
  cv2.randn(gauss_noise,128,150)
  gauss_noise=(gauss_noise*0.5).astype(np.uint8)

  gaus_image, gaus_ID=cv2.add(img_resize,gauss_noise), df["ID"].iloc[p]
  gaus_image = np.array(gaus_image)
  data_test = np.append(data_test, gaus_image)
  labels_test.append(gaus_ID)

  print(p)

  p += 1

print(data_test)
print(labels_test)

'''Save the results'''
np.save("data_test.npy", data_test)
np.save("labels_test.npy", labels_test)
