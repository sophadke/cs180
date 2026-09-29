import numpy as np 
import time 
import matplotlib.pyplot as plt 
import skimage.io as skio 
from skimage.transform import resize
from scipy.signal import convolve2d

D_x = np.array([[1,0,-1]])
D_y = np.array([[1],[0],[-1]])


def conv_four_loops(img, kernel): 

    
    kernel = kernel[::-1, ::-1] #flips the kernel over 
    kh, kw = kernel.shape
    #Handling odd lenghts 
    ph = kh // 2 
    pw = kw // 2 
    padded = np.pad(img, ((ph,ph), (pw,pw)))

    H, W = img.shape 
    out = np.zeros((H, W))

    for i in range(H): 
        for j in range(W): 
            total = 0 
            for u in range(kh): 
                for v in range(kw): 
                    total += padded[i + u, j + v] * kernel[u, v]
            out[i, j] = total 
    return out 


def conv_two_loops(img, kernel): 
    kernel = kernel[::-1, ::-1]
    kh, kw = kernel.shape 
    padded = np.pad(img,((kh // 2, (kh - 1) // 2), (kw // 2, (kw - 1) // 2)))

    H, W = img.shape 
    out = np.zeros((H, W))

    for i in range(H): 
        for j in range(W): 
            patch = padded[i:i + kh, j:j + kw]        
            out[i, j] = np.sum(patch * kernel)
    return out 


# Photo of myself 

img = skio.imread("me.jpg", as_gray=True)

height = 200 
width = int(200 * img.shape[1] / img.shape[0])
img = resize(img,(height, width))

plt.imsave("1_1_original.png", img, cmap="gray")


def comparsion_test(name, kernel): 
    start = time.time() 
    result_four = conv_four_loops(img, kernel)
    time_four = time.time() - start 

    start = time.time() 
    result_two = conv_two_loops(img, kernel)
    time_two = time.time() - start 

    start = time.time() 
    result_scipy = convolve2d(img, kernel, mode="same", boundary="fill")
    time_scipy = time.time() - start

    ### Same as scipy? 
    matched = np.allclose(result_four, result_scipy) and np.allclose(result_two, result_scipy)

    print(name)
    print("four loops:", round(time_four, 3), "seconds")
    print("two loops: ", round(time_two, 3), "seconds")
    print("scipy:     ", round(time_scipy, 5), "seconds")
    print("matches scipy:", matched)


    plt.imsave("1_1_" + name + ".png", result_two, cmap="gray")


box = np.ones((9,9)) / 81 

comparsion_test("box", box)

comparsion_test("Dx", D_x)

comparsion_test("Dy", D_y)


