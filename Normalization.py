if self.samplewise_center:
    x -= np.mean(x, keepdims=True)  # 减去每个批次feature的平均值实现0中心化
if self.samplewise_std_normalization:
    x /= (np.std(x, keepdims=True) + K.epsilon())  # 除以每个批次feature的标准差
if self.featurewise_center:
    self.mean = np.mean(x, axis=(0, self.row_axis,
                                 self.col_axis))  # 在底层为tendorflow时这里#self.row_axis=1，self.col_axis=2,即axis(0,1,2)。因为x是一个4维np,最后一维即图像的通道数，所
    # 以这里计算机的每一通道的像素平均值。
    broadcast_shape = [1, 1, 1]
    broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
    self.mean = np.reshape(self.mean, broadcast_shape)  # 对应mean的shape
    x -= self.mean  # 对每批次的数据减对应通道像素的均值

if self.featurewise_std_normalization:
    self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
    broadcast_shape = [1, 1, 1]
    broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
    self.std = np.reshape(self.std, broadcast_shape)
    x /= (self.std + K.epsilon())  # 对每批次的数据除以对应通道像素的标准差
ImageDataGenerator(
        featurewise_center=True,  #均值为0
        featurewise_std_normalization=True,#标准化处理
        samplewise_center=False,
        samplewise_std_normalization=False,
        )
