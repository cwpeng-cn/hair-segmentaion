# -*- encoding: utf-8 -*-

import os
from transform import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class HairMask(Dataset):
    def __init__(self, root_path, crop_size, mode='train', *args, **kwargs):
        super(HairMask, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val')
        self.mode = mode
        self.root_path = root_path
        self.info = self.get_list_info()

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(crop_size)
        ])

    def get_list_info(self):
        suffix = "jpg"
        mode = self.mode
        img_list, ann_list = [], []
        img_dir = os.path.join(self.root_path, "img_dir/" + mode)
        ann_dir = os.path.join(self.root_path, "ann_dir/" + mode)

        for name in sorted(os.listdir(img_dir)):
            if name.endswith(suffix):
                img_list.append(os.path.join(img_dir, name))
        for name in sorted((os.listdir(ann_dir))):
            if name.endswith(suffix):
                ann_list.append(os.path.join(ann_dir, name))
        return list(zip(img_list, ann_list))

    def __getitem__(self, idx):
        img_path, ann_path = self.info[idx]
        img = Image.open(img_path)
        label = Image.open(ann_path).convert('P')
        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label[label != 0] = 1
        return img, label

    def __len__(self):
        return len(self.info)


if __name__ == '__main__':
    dataset = HairMask(root_path="./data/HAIR", crop_size=(384, 384))
    for data in dataset:
        img, label = data
        print(label.min(), label.max())
