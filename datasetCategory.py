import os
import os.path
import json
import numpy as np


def pfnet_processing_dataset(self):
    with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
        train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        # train_ids是从train_test_split中取出来的训练的id，例如；3d2cb9d291ec39dc58a42593b26221da
    with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
        val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
    with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
        test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
    return train_ids, val_ids, test_ids


# TODO：需要去测试，这段代码是否正确
def pcn_processing_dataset(self):
    with open(os.path.join(self.root, 'shapenet', 'train.list'), 'r') as f:
        train_ids = set([str(line.strip().split('/')[1]) for line in f])

    with open(os.path.join(self.root, 'shapenet', 'valid.list'), 'r') as f:
        val_ids = set([str(line.strip().split('/')[1]) for line in f])

    with open(os.path.join(self.root, 'shapenet',  'test.list'), 'r') as f:
        test_ids = set([str(line.strip().split('/')[1]) for line in f])

    return train_ids, val_ids, test_ids

# def read_pcn_data():
#     seg = np.loadtxt('./test.pcd', skiprows=11).astype(np.float32)
#     print('777777======', seg)
#
#
# if __name__ == '__main__':
#     read_pcn_data()
