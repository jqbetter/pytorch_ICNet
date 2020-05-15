import os
import numpy as np
import torch
from config import args
import logging

def init_log(log_nm):
    #初始化程序日志配置
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    log_folder=args.log_dir
    if not os.path.exists(log_folder):
        os.makedev(log_folder)
    log_path=os.path.join(log_folder,log_nm)
    sh=logging.StreamHandler()
    fh=logging.FileHandler(log_path,mode="w")
    fh.setLevel(logging.DEBUG)
    formatter=logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def one_hot_it(label,label_info):
    semantic_map=[]
    for info in label_info:
        color=label_info[info]
        equality=np.equal(label,color)
        class_map=np.all(equality,axis=-1)
        semantic_map.append(class_map)
    return np.stack(semantic_map,axis=-1)

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=600, power=0.9):
	"""Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power
	"""
	if iter % lr_decay_iter or iter > max_iter:
	 	return optimizer
	lr = init_lr*(1 - iter/max_iter)**power
	optimizer.param_groups[0]['lr'] = lr
	return lr

def reverse_one_hot(image):
	"""
	Transform a 2D array in one-hot format (depth is num_classes),
	to a 2D array with only 1 channel, where each pixel value is
	the classified class key.

	# Arguments
		image: The one-hot format image

	# Returns
		A 2D array with the same width and height as the input, but
		with a depth size of 1, where each pixel value is the classified
		class key.
	"""
	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,1])

	# for i in range(0, w):
	#     for j in range(0, h):
	#         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
	#         x[i, j] = index
	image = image.permute(1, 2, 0)
	x = torch.argmax(image, dim=-1)
	return x

def colour_code_segmentation(image, label_values):
	"""
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,3])
	# colour_codes = label_values
	# for i in range(0, w):
	#     for j in range(0, h):
	#         x[i, j, :] = colour_codes[int(image[i, j])]
	label_values = [label_values[key] for key in label_values]
	colour_codes = np.array(label_values)
	x = colour_codes[image.astype(int)]
	return x

def compute_global_accuracy(pred, label):
	pred = pred.flatten()
	label = label.flatten()
	total = len(label)
	count = 0.0
	for i in range(total):
		if pred[i] == label[i]:
			count = count + 1.0
	return float(count) / float(total)

def fast_hist(a, b, n):
	'''
	a and b are predict and mask respectively
	n is the number of classes
	'''
	k = (a >= 0) & (a < n)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
	epsilon = 1e-5
	return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

def TN(y_true, y_predict):
    return np.sum((y_true == 0) & (y_predict == 0))

def FP(y_true, y_predict):
    return np.sum((y_true == 0) & (y_predict == 2))

def FN(y_true, y_predict):
    return np.sum((y_true == 2) & (y_predict == 0))

def TP(y_true, y_predict):
    return np.sum((y_true == 2) & (y_predict == 2))
'''  
     0:cloud 1:shadow 2:void
'''
def coutpixel(y_true, y_predict):
	N00=np.sum((y_true == 0) & (y_predict == 0))
	N11=np.sum((y_true == 1) & (y_predict == 1))
	N22=np.sum((y_true == 2) & (y_predict == 2))
	N01=np.sum((y_true == 0) & (y_predict == 1))
	N02=np.sum((y_true == 0) & (y_predict == 2))
	N10=np.sum((y_true == 1) & (y_predict == 0))
	N12=np.sum((y_true == 1) & (y_predict == 2))
	N20=np.sum((y_true == 2) & (y_predict == 0))
	N21=np.sum((y_true == 2) & (y_predict == 1))
	return N00,N11,N22,N01,N02,N10,N12,N20,N21

if __name__=="__main__":
    pass


