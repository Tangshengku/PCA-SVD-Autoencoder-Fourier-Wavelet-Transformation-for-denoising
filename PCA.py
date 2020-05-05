import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
import math

img = cv2.imread('D:/NoisedImages/Lena.jpg',1)
img1 = cv2.imread('D:/OriginalImages/Lena.jpg',1)#原图用以计算PSNR
b, g, r = cv2.split(img);#分离出图片的B，R，G颜色通道

#计算PSNR
def PSNR(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
# 如果噪音比较多，保留较少信息
#以下代码寻找保留信息最佳比例
"""a = np.zeros(shape=(500,1))
i=0.5
num=0
while(i<1):
   pca = PCA(i)  
   pca.fit(r)
   r2 = pca.transform(r)  # 将数据 r 降维，得到数据 r_low
   r_de = pca.inverse_transform(r2).astype(np.int)  # 将数据升到原来维度
   pca.fit(g)
   g2 = pca.transform(g)
   g_de = pca.inverse_transform(g2).astype(np.int)
   pca.fit(b)
   b2 = pca.fit_transform(b)
   b_de = pca.inverse_transform(b2).astype(np.int)
   img2 = cv2.merge([b_de, g_de, r_de])
   a[num]=PSNR(img2, img1)
   i+=0.005
   num+=1
print(np.argsort(-a,axis=0))"""

pca = PCA(0.875)
pca.fit(r)
r2 = pca.transform(r)  # 将数据 r 降维，得到数据 r_2
r_de = pca.inverse_transform(r2).astype(np.int)  # 将数据升到原来维度
pca.fit(g)
g2 = pca.transform(g)
g_de = pca.inverse_transform(g2).astype(np.int)
pca.fit(b)
b2 = pca.fit_transform(b)
b_de = pca.inverse_transform(b2).astype(np.int)
img2 = cv2.merge([b_de, g_de, r_de])

print(PSNR(img2, img1))
cv2.imwrite('D:/Lena_de.jpg',img2)
