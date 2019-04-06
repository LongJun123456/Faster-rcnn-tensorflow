import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import config as cfg

class pascal_voc(object):
    def __init__(self, name, phase, fliped, rebuild=False):
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')   
        #self.data_path = os.path.join(self.devkil_path)  
        self.cache_path = cfg.CACHE_PATH    
        self.batch_size = cfg.BATCH_SIZE    
        self.target_size = cfg.target_size  
        self.max_size = cfg.max_size    
        self.classes = cfg.CLASSES  #['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus'....]
        self.pixel_means = cfg.PIXEL_MEANS  
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))   
        self.flipped = fliped  
        self.phase = phase  
        self.name = name
        self.rebuild = rebuild  
        self.cursor = 0    
        self.epoch = 1     
        #self.gt_labels = None
        self.gt_labels = self.combine_imdb(name)
        self.num_gtlabels = len(self.gt_labels)
    
    def combine_imdb(self, name):
        """ combine VOC2007 imdb and VOC2012 imdb"""
        gt_labels = [self.prepare(imdb_name) for imdb_name in name.split('+')]
        gt_label = gt_labels[0]
        if len(gt_labels) > 1:
            for g in gt_labels[1:]:
                gt_label.extend(g)
        return gt_label

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) 
        if flipped:
            image = image[:, ::-1, :]
        return image
    
    def get(self):
        """ get the batch train/test data"""
        count = 0
        tf_blob = {}
        assert self.batch_size == 1, "only support single batch" 
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            image = self.image_read(imname, flipped=flipped)
            image, image_scale = self.prep_im_for_blob(image, self.pixel_means, self.target_size, self.max_size)
            image = np.reshape(image, (self.batch_size, image.shape[0], image.shape[1], 3)) 
            gt_box = self.gt_labels[self.cursor]['boxes'] * image_scale 
            gt_cls = self.gt_labels[self.cursor]['gt_classs']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        tf_blob = {'image':image, 'scale':image_scale, 'cls':gt_cls, 'box': gt_box, 'imname': imname}
        return tf_blob #image.shape=[batch,size,size,3] image_scale, gt_box.shape=[num_objs,4]

    

    def prepare(self, imdb_name):
        """prepare the imdb, if pkl exists, you can load the pkl file"""
        gt_labels = self.load_labels(imdb_name)
        if self.flipped:
            print('Appending horizontally-flipped training examples ...') #{'boxes':boxes, 'gt_classs':gt_classes, 'imname':imname}
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                width_pre = copy.deepcopy(gt_labels_cp[idx]['boxes'][:,0])
                gt_labels_cp[idx]['boxes'][:,0] = gt_labels_cp[idx]['image_size'][0] - gt_labels_cp[idx]['boxes'][:,2]
                gt_labels_cp[idx]['boxes'][:,2] = gt_labels_cp[idx]['image_size'][0] - width_pre
#                gt_labels_cp[idx]['boxes'][:,[0,2]] = gt_labels_cp[idx]['image_size'][0] - gt_labels_cp[idx]['boxes'][:,[0,2]][:,::-1]
            gt_labels += gt_labels_cp
        if self.phase == 'train':
            np.random.shuffle(gt_labels)
        #self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self, imdb_name):
        cache_file_tmp = os.path.join(self.cache_path, imdb_name)
        if not os.path.exists(cache_file_tmp):
            os.mkdir(cache_file_tmp)
        cache_file = os.path.join(cache_file_tmp, 'pascal_' + self.phase + '_gt_labels.pkl')
        
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)  
            return gt_labels

        print('Processing gt_labels from: ' + self.devkil_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':
            txtname = os.path.join(
                self.devkil_path, imdb_name, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(
                self.devkil_path, imdb_name, 'ImageSets', 'Main', self.phase + '.txt')
            #self.flipped = False
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.image_index:
            gt_label = self.load_pascal_annotation(index, imdb_name) 
            gt_labels.append(gt_label)
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index, imdb_name):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self.devkil_path, imdb_name, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        image_size = tree.find('size')
        size_info = np.zeros((2,), dtype=np.float32)
        size_info[0] = float(image_size.find('width').text)
        size_info[1] = float(image_size.find('height').text)
        num_objs = len(objs) 
        boxes = np.zeros((num_objs, 4), dtype=np.float32) 
        gt_classes = np.zeros((num_objs), dtype=np.int32) 
        difficult = np.empty((num_objs))
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self.class_to_ind[obj.find('name').text.lower().strip()] 
            boxes[ix, :] = [x1, y1, x2, y2] 
            gt_classes[ix] = cls 
            imname = os.path.join(self.devkil_path, imdb_name, 'JPEGImages', index + '.jpg')
            difficult[ix] = int(obj.find('difficult').text)
        return {'boxes':boxes, 'gt_classs':gt_classes, 'imname':imname, 'flipped':False, 'image_size':size_info, 'image_index': index, 'diff': difficult}
    
    def prep_im_for_blob(self, im, pixel_means, target_size, max_size): 
            """image data process: de-contextualization and resize"""
            im = im.astype(np.float32, copy=False)
            im -= pixel_means 
            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min) 
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,interpolation=cv2.INTER_LINEAR)
            return im, im_scale


    def voc_ap(self, rec, prec): 
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
        
        i = np.where(mrec[1:] != mrec[:-1])[0] 
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]) 
        
        return ap
        
if __name__ == '__main__':
    pascal = pascal_voc('test')
    tf_blob = pascal.get()
    print (len(pascal.gt_labels))
    #print (len(pascal.gt_labels))
