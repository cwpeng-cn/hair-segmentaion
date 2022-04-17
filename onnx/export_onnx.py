import os
import sys

sys.path.append("..")

import torch
import argparse
from model import BiSeNet
from torch.autograd import Variable

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', default="../data/checkpoint/79999_iter.pth", type=str,
                        required=False, help='path of the checkpoint that will be converted')
    parser.add_argument('--output-path', default="./hair.onnx", type=str,
                        required=False, help='path for saving the ONNX model')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.ckpt_path):
        print('Cannot find checkpoint path: {0}'.format(args.ckpt_path))
        exit()

    # define model & load checkpoint
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    state_dict = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict)
    net.eval()

    # prepare dummy_input
    batch_size = 1
    height = 512
    width = 512
    dummy_input = Variable(torch.randn(batch_size, 3, height, width))

    # export to onnx model
    torch.onnx.export(
        net, dummy_input, args.output_path, export_params=True, opset_version=11,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'output': {0: 'batch_size', 2: 'height', 3: 'width'}})
