#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import cv2
import torch
import numpy as np
import os.path as osp
from PIL import Image
from model import BiSeNet
import torchvision.transforms as transforms


def evaluate(respth, dspth='./data', cp='model_final_diss.pth'):
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    save_pth = osp.join(respth, cp)
    net.load_state_dict(torch.load(save_pth, map_location='cpu'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        for image_path in os.listdir(dspth):
            if image_path != ".DS_Store":
                img = Image.open(osp.join(dspth, image_path))
                image = img.resize((512, 512), Image.BILINEAR)
                # image=img
                img = to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img
                out = net(img)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)
                print(image_path)
                print(parsing.max(), parsing.min())
                print(np.unique(parsing))
                parsing[parsing == 1] = 0
                parsing[parsing == 17] = 1
                parsing[parsing != 1] = 0
                cv2.imshow("hah", parsing.astype(np.uint8) * 255)
                cv2.waitKey(0)


if __name__ == "__main__":
    evaluate(respth='./data/checkpoint', dspth='/Users/cwpeng/Desktop/image', cp='79999_iter.pth')
