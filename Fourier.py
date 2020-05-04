import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
#高斯低通滤波器
def gaussian_make_transform(d,image_size):
    trans_matrix=np.zeros(image_size)
    center_point = tuple(map(lambda x: (x - 1) / 2, trans_matrix.shape))
    for i in range(trans_matrix.shape[0]):
        for j in range(trans_matrix.shape[1]):
            dis=math.sqrt((center_point[0]-i)**2+(center_point[1]-j)**2)
            trans_matrix[i,j]=np.exp(-(dis**2)/(2*(d**2)))
    return  trans_matrix

def gaussian_low_fiter(image,d):
    D=gaussian_make_transform(d,image.shape)
    image_fft = np.fft.fft2(image)
    image_shift = np.fft.fftshift(image_fft)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(image_shift * D)))
    return new_img
#butterworth低通滤波器
def butterworth_make_transform(d,image_size,n):
    trans_matrix = np.zeros(image_size)
    center_point = tuple(map(lambda x: (x - 1) / 2, trans_matrix.shape))
    for i in range(trans_matrix.shape[0]):
        for j in range(trans_matrix.shape[1]):
            dis = math.sqrt((center_point[0] - i) ** 2 + (center_point[1] - j) ** 2)
            trans_matrix[i, j] = 1-1/((1+(d/dis))**n)
    return trans_matrix

def butterworth_low_filter(image,d,n):
    D = butterworth_make_transform(d, image.shape,n)
    image_fft = np.fft.fft2(image)
    image_shift = np.fft.fftshift(image_fft)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(image_shift * D)))
    return new_img
#峰值信噪比
def psnr2(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

image=cv2.imread("noisedimages/Lena.jpg")
image_original=cv2.imread("OriginalImages/Lena.jpg")
b,g,r=cv2.split(image)
b=gaussian_low_fiter(b/1.0,25)
g=gaussian_low_fiter(g/1.0,25)
r=gaussian_low_fiter(r/1.0,25)
#b=butterworth_low_filter(b/1.0,20,2)
#g=butterworth_low_filter(g/1.0,20,2)
#r=butterworth_low_filter(r/1.0,20,2)
image_plt=cv2.merge([r,g,b])
image_cv=cv2.merge([b,g,r])#cv和plt两者的通道组织方式相反
for i in range(image.shape[2]):
    print(psnr2(image_cv[:,:,i],image_original[:,:,i]))
plt.imshow(image_plt/255)
plt.show()
