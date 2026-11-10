import matplotlib.pyplot as plt 
import numpy as np 

im1 = plt.imread("../images/bank_01.jpg")
im2 = plt.imread("../images/bank_03.jpg")

plt.figure(figsize=(10, 8))
plt.imshow(im1)
plt.title("Bank 01")
pts1 = np.array(plt.ginput(8, timeout=0))
plt.close()

plt.figure(figsize=(10, 8))
plt.imshow(im2)
plt.title("Bank 03")
pts2 = np.array(plt.ginput(8, timeout=0))
plt.close()

np.savez("bank_01_03_points.npz", pts1=pts1, pts2=pts2)
print("connection points")