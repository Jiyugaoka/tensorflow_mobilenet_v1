# -*- coding: UTF-8 -*-

import os
import argparse
import numpy as np
import cv2 as cv
from lxml.etree import Element,SubElement,ElementTree, parse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="the path of your dataset")
parser.add_argument("-n", "--nclass", help="the number of how many classes you want to classify")



class Dataset(object):

    def __init__(self, nb_classes, train_dataset_mean=None):
        self.nb_classes = nb_classes
        self.train_dataset_mean = train_dataset_mean

    def make_dir(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir

    def write_xml(self, inImgDir, outAnnoDir):
        trainImgs = sorted(os.listdir(inImgDir))
        # trainAnnos = sorted(os.listdir(outAnnoDir))
        for index, imageClass in enumerate(trainImgs):
            subDirIn = os.path.join(inImgDir, imageClass)
            subdirOut = self.make_dir(os.path.join(outAnnoDir, imageClass))
            imageNames = os.listdir(subDirIn)
            for imageName in imageNames:
                image_filename = os.path.join(subDirIn, imageName)
                img = cv.imread(image_filename)
                height, width, depth = img.shape
                annoName = imageName.split('.')[0] + '.xml'

                print('Writing annotation:%s' % annoName)

                node_root = Element('annotation')

                node_filename = SubElement(node_root, 'filename').text = image_filename
                node_size = SubElement(node_root, "size")

                node_width = SubElement(node_size, 'width').text = str(height)
                node_height = SubElement(node_size, 'height').text = str(width)
                node_depth = SubElement(node_size, 'depth').text = '3'

                node_label = SubElement(node_root, "label").text = "%s, %s" % (str(index), imageClass)

                doc = ElementTree(node_root)
                doc.write(open(os.path.join(subdirOut, annoName), "wb+"), pretty_print=True)

    def read_xml(self, annoDir):
        # give a annodir,and return a dict,each key is the imageclass, its value is all the imageInfo in this dir,value type is list.
        datasetInfo = {}
        subDirList = sorted(os.listdir(annoDir))
        for imageClass in subDirList:
            datasetInfo[imageClass] = []
            subAnnoDir = os.path.join(annoDir, imageClass)
            subAnnoList = sorted(os.listdir(subAnnoDir))
            for subAnno in subAnnoList:
                imgInfo = self.parse_xml(os.path.join(subAnnoDir, subAnno))
                datasetInfo[imageClass].append(imgInfo)
        return datasetInfo

    def parse_xml(self, file_path):
        imgInfo = {}
        tree = parse(file_path)
        for ele in tree.iter():
            if 'filename' in ele.tag:
                imgInfo['filename'] = ele.text
            if 'width' in ele.tag:
                imgInfo['width'] = ele.text
            if 'height' in ele.tag:
                imgInfo['height'] = ele.text
            if 'depth' in ele.tag:
                imgInfo['depth'] = ele.text
            if 'label' in ele.tag:
                imgInfo['label'] = ele.text
        return imgInfo

    def labels_to_one_hot(self, y, nb_classes):
        # y is the class label
        Y = np.zeros(nb_classes)
        Y[y] = 1.
        return Y

    def find_file_abs_path(self, dir_in):
        # give a dir, return all absolute path of the items in this dir.
        img_dir_output = []
        imgs = sorted(os.listdir(dir_in))
        for img in imgs:
            img_dir_path = os.path.join(dir_in, img)
            img_dir_output.append(img_dir_path)
        return img_dir_output

    def compute_pixel_mean(self, img_dir):

        img_dir_lists = [os.path.join(img_dir, className) for className in sorted(os.listdir(img_dir))]
        image_paths = []

        for img_dir in img_dir_lists:
            image_paths.extend(self.find_file_abs_path(img_dir))
        image_basenames = [os.path.basename(img).split('.')[0] for img in image_paths]
        count = len(image_paths)
        mean_B, mean_G, mean_R = .0, .0, .0
        # std_B, std_G, std_R = .0, .0, .0
        for i, image_path in enumerate(image_paths):
            print('processing image %s' % str(image_basenames[i]))
            image = cv.imread(image_path)
            mean_B += np.mean(image[:, :, 0]) / count
            mean_G += np.mean(image[:, :, 1]) / count
            mean_R += np.mean(image[:, :, 2]) / count
        self.train_dataset_mean = [mean_B, mean_G, mean_R]
        print('The pixel means of this dataset is (%5.2f, %5.2f, %5.2f)' % (mean_B, mean_G, mean_R))
        # The pixel means of this dataset is (107.94, 115.34, 118.34)
        # return mean_B, mean_G, mean_R

    def normalize_image_by_chanel(self, image, dataset_mean_BGR):
        new_image = np.zeros(image.shape)
        for chanel in range(3):
            # mean = np.mean(image[:, :, chanel])
            # std = np.std(image[:, :, chanel])
            # new_image[:, :, chanel] = (image[:, :, chanel] - mean) / std
            new_image[:, :, chanel] = image[:, :, chanel] - dataset_mean_BGR[chanel]
        return new_image

    def read_data_label(self, data_dir, label):
        # input:
        #      data_dir:  [type: list, content: each image_dir used for cv2.imread()]
        #      label: [type: list, label[0]: a string type number of the label, label[1]: the name]
        # output:
        #       data: image data
        #       labelInfo: one hot label, int(label[0]), label[1]
        data, labelInfo = [], []
        for ind, img_dir in enumerate(data_dir):
            img_raw = cv.imread(img_dir)
            new_image = self.normalize_image_by_chanel(img_raw, self.train_dataset_mean)
            data.append(new_image)
            Y = self.labels_to_one_hot(np.int(label[ind][0]), self.nb_classes)
            labelInfo.append([Y, np.int(label[ind][0]), label[ind][1]])
        return data, labelInfo

    def label_shuffle(self, annoDir):
        # input: Comma-separated list of train image dir
        # output:label shuffled image path used for cv2.imread()
        anno_dir_lists = [os.path.join(annoDir, className) for className in sorted(os.listdir(annoDir))]
        # anno_paths is a list, its sublist is all the absolute path of the item in it.
        anno_paths = []
        for anno_dir in anno_dir_lists:
            anno_paths.append(self.find_file_abs_path(anno_dir))

        label_nums = [len(anno_path) for anno_path in anno_paths]
        maxNum = max(label_nums)
        print('max sample number is %d' % (maxNum))
        index = list(range(maxNum))

        # image_paths and image_labels have the same structure with anno_paths.
        image_paths = []
        image_labels = []
        for anno_path in anno_paths:
            # image_infos is a list, each items in it is the imginfo of image_xml file with respect to.
            image_infos = [self.parse_xml(anno_dir) for anno_dir in anno_path]
            image_paths.append([items['filename'] for items in image_infos])
            image_labels.append(
                [[items['label'].split(',')[0].strip(), items['label'].split(',')[1].strip()] for items in image_infos])

        # shuffled_image and shuffed_label is a list,the items in it are the shuffled image path and [label_num, label_text] respectively.
        shuffled_image = []
        shuffed_label = []
        for i in range(len(anno_dir_lists)):
            np.random.shuffle(index)
            image_path = np.array(image_paths[i])
            image_label = np.array(image_labels[i])
            shuffled_index = [ind % label_nums[i] for ind in index]
            shuffled_image.extend(image_path[shuffled_index])
            shuffed_label.extend(image_label[shuffled_index])

        totalNum = len(shuffled_image)
        finalIndex = list(range(totalNum))
        np.random.shuffle(finalIndex)

        return np.array(shuffled_image)[finalIndex], np.array(shuffed_label)[finalIndex]

    def get_input(self, anno_dir):
        # input : dataset anno dir
        # outPut : dataset imagedir used for cv2.imread()
        # and image label with (label_index, label_name)
        print('Loading data...')
        # data_dir and label have the same type as the shuffled_image and shuffed_label
        data_dir, label = [], []
        # type(dataset_info) = {}
        dataset_info = self.read_xml(anno_dir)
        for key in dataset_info:
            # type(classInfo) = [],and the items in it is all the infomation of the respective image, type is a dict.
            classInfo = dataset_info[key]
            # type(imageInfo) = {}
            for imageInfo in classInfo:
                data_dir.append(imageInfo['filename'])
                label_index = imageInfo['label'].split(',')[0].strip()
                label_name = imageInfo['label'].split(',')[1].strip()
                label.append([label_index, label_name])

        return np.array(data_dir), np.array(label)

    def batch_iter(self, data, label, batch_size, num_epochs, shuffle=True):

        data_size = len(data)
        label_size = len(label)
        assert data_size == label_size, 'The data_size do not match label_size!!!'
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_label = label[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_label = label
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                batch_data, batch_label = shuffled_data[start_index:end_index], shuffled_label[start_index:end_index]

                yield self.read_data_label(batch_data, batch_label)
def _main_(args):
    dataset_dir = args.dataset
    n_class = args.nclass
    dataset = Dataset(n_class)

    print('Creating train dataset annotation.')
    trainImgDir = os.path.join(dataset_dir, 'train')
    trainOutAnnoDir = os.path.join(dataset_dir, 'trainAnno')
    dataset.write_xml(trainImgDir, trainOutAnnoDir)

    print('Creating test dataset annotation.')
    testImgDir = os.path.join(dataset_dir, 'test')
    testOutAnnoDir = os.path.join(dataset_dir, 'testAnno')
    dataset.write_xml(testImgDir, testOutAnnoDir)

    dataset.compute_pixel_mean(trainImgDir)

    if __name__ == '__main__':
        args = parser.parse_args()
        _main_(args)

        """
        trainData, trainLabel = get_input(dataset_dir)
        for index, data in enumerate(trainData):
            print("image path: %s, image label[0]: %s, image label[1]: %s"%(data, trainLabel[index][0], trainLabel[index][1]), type(trainLabel[index][0]))
            #image path: D:\ImageProcessing\carDetection\VOC2005\VOC2005Train\Person\person_306_croped.jpg, image label: ['3' 'Person'] <class 'numpy.ndarray'>
        print(len(trainData))#result is 2162

        shuffled_trainData, shuffled_trainLabel = label_shuffle(dataset_dir)#max sample number is 820
        for index, data in enumerate(shuffled_trainData):
            print("image path: %s, image label: %s" % (data, shuffled_trainLabel[index]), type(shuffled_trainLabel[index]))
            #image path: D:\ImageProcessing\carDetection\VOC2005\VOC2005Train\Car\carsgraz_214_croped_hfliped.jpg, image label: ['1' 'Car'] <class 'numpy.ndarray'>
        print(len(shuffled_trainData))# 820*4=3280
        """




    



    
