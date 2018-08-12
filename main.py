import os
import pickle
import h5py

import numpy as np
from PIL import Image
from scipy.misc import imresize


def main():
	dataset_path = '/media/burak/Data/ILSVRC Classification'
	image_size = 224
	# mean is in BGR
	mean_intensity = np.array([103.86645, 116.78735 , 123.690605], dtype=np.float32)

	# read metadata
	with open ('ilsvrc_meta', 'rb') as fp:
		ids_out, names_out, wnids_out, val_gt_out = pickle.load(fp)
	
	# read "correct" WNID order
	with open ('synset_words.txt', 'r') as f:
		lines = f.readlines()
	WNIDs = []
	for line in lines:
		WNIDs.append(line.split()[0])

	# correct validation labels
	val_labels = []
	for gt in val_gt_out:
		WNID = wnids_out[gt - 1]
		val_labels.append(WNIDs.index(WNID))

	# correct class name order
	class_names = []
	for WNID in WNIDs:
		target_ind = wnids_out.index(WNID)
		class_names.append(names_out[target_ind])
	no_classes = len(class_names)

	# create hdf5 file
	f = h5py.File('ilsvrc2012.h5', 'w')
	dt_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
	dt_str = h5py.special_dtype(vlen=str)

	# write class names
	class_names_h = f.create_dataset('class_names', (len(names_out),), dtype=dt_str)
	for ind_class_name, class_name in enumerate(class_names):
		class_names_h[ind_class_name] = class_name

	# write data types
	data_types = ['train', 'val']
	data_types_h = f.create_dataset('data_types', (len(data_types),), dtype=dt_str)
	for ind_data_type, data_type in enumerate(data_types):
		data_types_h[ind_data_type] = data_type

	# write dataset mean
	f.create_dataset('mean', data=mean_intensity)

	# put training data in lists
	image_paths = []
	image_names = []
	labels = []
	for ind_WNID, WNID in enumerate(WNIDs):
		file_names = os.listdir(dataset_path + '/ILSVRC2012_img_train/' + WNID)
		for file_name in file_names:
			image_path = dataset_path + '/ILSVRC2012_img_train/' + WNID + '/' + file_name
			image_paths.append(image_path)
			image_names.append(file_name)
			labels.append(ind_WNID)

	#allocate space for training images
	no_train = len(image_paths)
	image_h = f.create_dataset('train_images', (no_train,), dtype=dt_uint8)
	name_h = f.create_dataset('train_image_names', (no_train,), dtype=dt_str)
	label_h = f.create_dataset('train_labels', (no_train, no_classes), dtype=np.int)

	# training data is to be shuffled
	inds_perm = np.random.permutation(no_train)

	for ind in range(no_train):
		if ind % 10000 == 0:
			print str(ind) + '/' + str(no_train)
		image = Image.open(image_paths[inds_perm[ind]])
		# remove any alpha channels, etc.
		image = image.convert('RGB')
		# resize so that smaller dimension is image_size
		size_lower = float(min(image.size))
		new_size = (int(round(image.size[0] / size_lower * image_size)), int(round(image.size[1] / size_lower * image_size)))
		image = image.resize(new_size, resample=Image.BILINEAR)
		np_image = np.array(image)
		np_image = np_image[(np_image.shape[0] - image_size) / 2 :(np_image.shape[0] + image_size) / 2,
							(np_image.shape[1] - image_size) / 2 :(np_image.shape[1] + image_size) / 2]
		# RGB -> BGR
		np_image = np_image[:, :, ::-1]
		# HDF5 takes 1D data
		np_image = np_image.flatten()

		# write to HDF5
		image_h[ind] = np_image
		name_h[ind] = image_names[inds_perm[ind]]
		label_h[ind, labels[inds_perm[ind]]] = 1

	# do the same for validation data
	image_names = os.listdir(dataset_path + '/ILSVRC2012_img_val')
	image_names.sort()
	labels = val_labels
	no_val = len(image_names)

	inds_perm = np.random.permutation(no_val)

	image_h = f.create_dataset('val_images', (no_val,), dtype=dt_uint8)
	name_h = f.create_dataset('val_image_names', (no_val,), dtype=dt_str)
	label_h = f.create_dataset('val_labels', (no_val, no_classes), dtype=np.int)

	for ind in range(no_val):
		if ind % 10000 == 0:
			print str(ind) + '/' + str(no_val)
		image = Image.open(dataset_path + '/ILSVRC2012_img_val/' + image_names[inds_perm[ind]])
		# remove any alpha channels, etc.
		image = image.convert('RGB')
		# resize so that smaller dimension is image_size
		size_lower = float(min(image.size))
		new_size = (int(round(image.size[0] / size_lower * image_size)), int(round(image.size[1] / size_lower * image_size)))
		image = image.resize(new_size, resample=Image.BILINEAR)
		np_image = np.array(image)
		np_image = np_image[(np_image.shape[0] - image_size) / 2 :(np_image.shape[0] + image_size) / 2,
							(np_image.shape[1] - image_size) / 2 :(np_image.shape[1] + image_size) / 2]
		# RGB -> BGR
		np_image = np_image[:, :, ::-1]
		# HDF5 takes 1D data
		np_image = np_image.flatten()

		# write to HDF5
		image_h[ind] = np_image
		name_h[ind] = image_names[inds_perm[ind]]
		label_h[ind, labels[inds_perm[ind]]] = 1

	f.close()

	# show random images to test
	f = h5py.File('ilsvrc2012.h5', 'r')
	class_names_h = f['class_names']
	data_types_h = f['data_types']
	while True:
		ind_data_type = np.random.randint(0, len(data_types_h))
		data_type = data_types_h[ind_data_type]

		image_h = f[data_type + '_images']
		name_h = f[data_type + '_image_names']
		label_h = f[data_type + '_labels']

		ind_image = np.random.randint(0, len(image_h))
		np_image = np.reshape(image_h[ind_image], [image_size, image_size, 3])
		np_image = np_image[:, :, ::-1]
		image = Image.fromarray(np_image, 'RGB')
		image.show()

		print('Image type: ' + data_type)
		print('Image name: ' + name_h[ind_image])
		for ind_class_name, class_name in enumerate(class_names_h):
			if label_h[ind_image][ind_class_name] == 1:
				print class_name
		raw_input("...")


if __name__ == '__main__':
	main()