
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import math

# config
image_row=350
image_col=500
istrain=True
## Denoising
def psnr2(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
inputs_ = tf.placeholder(tf.float32, (None, image_row, image_col, 3), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, image_row, image_col, 3), name='targets')

### 编码器（不同的图像参数得到的结果不同，下面的注释以Lena的350*500为例）
#inputs_=tf.layers.batch_normalization(inputs_,training=istrain)
conv1 = tf.layers.conv2d(inputs_, 32, (3, 3), padding='same', activation=tf.nn.relu)
# 当前shape: 350x500x32
maxpool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same')
# 当前shape: 175x250x32
#maxpool1=tf.layers.batch_normalization(maxpool1,training=istrain)
conv2 = tf.layers.conv2d(maxpool1, 32, (3, 3), padding='same', activation=tf.nn.relu)
# 当前shape: 175x250x32
maxpool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='same')
# 当前shape: 88x125x32
#maxpool2=tf.layers.batch_normalization(maxpool2,training=istrain)
conv3 = tf.layers.conv2d(maxpool2, 16, (3, 3), padding='same', activation=tf.nn.relu)
# 当前shape: 88x125x16
encoded = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding='same')
# 当前shape: 63x44x16
#encoded=tf.layers.conv2d(encoded,16,(1,1),padding='same',activation=tf.nn.relu)

### 解码器

upsample1 = tf.image.resize_nearest_neighbor(encoded, (np.uint8(image_row/4), np.uint8(image_col/4)))
# 当前shape: 125x88x16
#upsample1=tf.layers.batch_normalization(upsample1,training=istrain)
conv4 = tf.layers.conv2d(upsample1, 16, (3, 3), padding='same', activation=tf.nn.relu)

# 当前shape: 125x88x16
upsample2 = tf.image.resize_nearest_neighbor(conv4, (np.uint8(image_row/2), np.uint8(image_col/2)))
# 当前shape: 250x175x16
#upsample2=tf.layers.batch_normalization(upsample2,training=istrain)
conv5 = tf.layers.conv2d(upsample2, 32, (3, 3), padding='same', activation=tf.nn.relu)
# 当前shape: 250x175x32
upsample3 = tf.image.resize_nearest_neighbor(conv5, (image_row, image_col))
# 当前shape: 500x350x32
#upsample3=tf.layers.batch_normalization(upsample3,training=istrain)
conv6 = tf.layers.conv2d(upsample3, 32, (3, 3), padding='same', activation=tf.nn.relu)
# 当前shape: 500x350x32


logits = tf.layers.conv2d(conv6, 3, (3, 3), padding='same', activation=None)
# 当前shape: 500x350x1


decoded = tf.nn.sigmoid(logits, name='decoded')
#L2 = tf.nn.l2_loss(conv1) + tf.nn.l2_loss(conv2) + tf.nn.l2_loss(conv3) +tf.nn.l2_loss(conv4)+tf.nn.l2_loss(conv5)+tf.nn.l2_loss(conv6)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)#+0.01*L2
cost = tf.reduce_mean(loss)
##其他的优化方法，可以尝试，也许很有用
#opt=tf.train.GradientDescentOptimizer(0.001).minimize(cost)
opt = tf.train.AdamOptimizer(0.001).minimize(cost)
sess = tf.Session()


epochs = 30


sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(1,50):
        noisy_imgs=cv2.imread('train/Lena/Lena'+str(ii)+'.jpg')
        imgs = np.zeros(noisy_imgs.shape, dtype=np.float32)
        ##读取噪声图像并图像归一化
        cv2.normalize(noisy_imgs, imgs, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        ##将图像转换成输入格式
        imgs=imgs.reshape((1,image_row, image_col,3))
        ##Ground Truth
        gt=cv2.imread('OriginalImages/Lena.jpg')
        gt_norm=np.zeros(gt.shape,dtype=np.float32)
        cv2.normalize(gt, gt_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        gt_norm=gt_norm.reshape((1,image_row, image_col,3))

        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,targets_: gt_norm})
    ##每一整次训练输出损失
    print("Epoch: {}/{}...".format(e + 1, epochs),
         "Training loss: {:.4f}".format(batch_cost))

noisy_imgs=cv2.imread('test/Lena.jpg')
#noisy_imgs=cv2.resize(noisy_imgs,(500,350))
#noisy_imgs = np.clip(noisy_imgs, 0., 1.)
noisy_imgs_norm=np.zeros(noisy_imgs.shape,dtype=np.float32)
cv2.normalize(noisy_imgs, noisy_imgs_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#with tf.Session() as sess2:
 #   noisy_imgs = noisy_imgs.eval()
reconstructed = sess.run(decoded, feed_dict={inputs_: noisy_imgs_norm.reshape((1, image_row, image_col, 3))})
reconstructed=np.uint8(reconstructed*255.0)
reconstructed=reconstructed.reshape(image_row, image_col,3)
cv2.imshow("0",reconstructed)
cv2.waitKey()
gt=cv2.imread('OriginalImages/Lena.jpg')
print(psnr2(reconstructed,gt))

