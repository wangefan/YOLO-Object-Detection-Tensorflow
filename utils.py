import os
import json
import numpy as np
import cv2
import pickle
import copy
import settings

class DataSet:
    def __init__(self, phase):
        self.data_path = settings.data_set_path
        self.cache_file_path = settings.cache_file_path
        self.image_size = settings.image_size
        self.cell_size = settings.cell_size
        self.classes = settings.classes_name
        self.class_to_ind = settings.classes_dict
        self.flipped = settings.flipped
        self.phase = phase
        self.gt_labels = None
        self.prepare()

    def image_read(self, imname, flipped = False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in xrange(self.cell_size):
                    for j in xrange(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - 1 - gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    """ 根據data_set_path讀取image與json，會先檢查cache_file_path有沒有cache
      Args:
      Returns:
        gt_labels: [
                    {'imname': imname, 'label': label, 'flipped': False},
                    {'imname': imname, 'label': label, 'flipped': False}
                    ..
                   ]
        imname = '{data_set_path}/{artificial_train|artificial_test}/img/xxx.png'
        注意label是tensor(cell, cell, 5 + num_class) => 第三維內容[0|1(該格是否有物件), x_center, y_center, w, h, ...(num_class)]
    """
    def load_labels(self):
        cache_file = os.path.join(self.cache_file_path ,'cache_labels.pkl')
        if os.path.isfile(cache_file):
            print('Loading gt_labels from cache: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)
        image_names = os.listdir(os.path.join(self.get_label_path(), 'img'))
        image_names = [os.path.splitext(name)[0] for name in image_names] # 去除附檔名
        gt_labels = []
        for name in image_names:
            label, num = self.load_annotation(name)
            if num == 0:
                continue
            imname = os.path.join(self.get_label_path(), 'img', name + '.png')
            gt_labels.append({'imname': imname, 'label': label, 'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        if not os.path.exists(self.cache_file_path):
            os.makedirs(self.cache_file_path)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    """ 根據name讀取json
          Args:
            name:   檔名(不包含副檔名
            
          Returns:
            label:  tensor(cell, cell, 5 + num_class) => 第三維內容[0|1(該格是否有物件), x_center, y_center, w, h, ...(num_class)]
            num:      
        """
    def load_annotation(self, name):
        imname = os.path.join(self.get_label_path(), 'img', name + '.png')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]

        label = np.zeros((self.cell_size, self.cell_size, 5 + settings.num_class))
        filename = os.path.join(self.get_label_path(), 'ann', name + '.json')
        with open(filename) as data_file:
            data = json.load(data_file)
            objs = data["objects"]
            for obj in objs:
                xmin = float(min(obj["points"]["exterior"][0][0], obj["points"]["exterior"][1][0]))
                ymin = float(min(obj["points"]["exterior"][0][1], obj["points"]["exterior"][1][1]))
                xmax = float(max(obj["points"]["exterior"][0][0], obj["points"]["exterior"][1][0]))
                ymax = float(max(obj["points"]["exterior"][0][1], obj["points"]["exterior"][1][1]))
                x1 = max(min(xmin * w_ratio, self.image_size), 0)
                y1 = max(min(ymin * h_ratio, self.image_size), 0)
                x2 = max(min(xmax * w_ratio, self.image_size), 0)
                y2 = max(min(ymax * h_ratio, self.image_size), 0)
                cls_ind = self.class_to_ind[obj['classTitle'].lower().strip()]
                boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
                x_ind = int(boxes[0] * self.cell_size / self.image_size)
                y_ind = int(boxes[1] * self.cell_size / self.image_size)
                if label[y_ind, x_ind, 0] == 1:
                    continue
                label[y_ind, x_ind, 0] = 1
                label[y_ind, x_ind, 1:5] = boxes
                label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)

    """ 根據setting.phase得到img與ann所在的path
              Args:

              Returns:
                dest_path:  底下包含"img"與"ann"所在的路徑
    """
    def get_label_path(self):
        if self.phase == 'train':
            return os.path.join(self.data_path, 'artificial_train')
        else:
            return os.path.join(self.data_path, 'artificial_test')