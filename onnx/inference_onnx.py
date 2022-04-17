"""
Inference ONNX model of MODNet

Arguments:
    --image-path: path of the input image (a file)
    --output-path: path for saving the predicted alpha matte (a file)
    --model-path: path of the ONNX model

Example:
python inference_onnx.py \
    --image-path=demo.jpg --output-path=matte.png --model-path=modnet.onnx
"""

import os
import cv2
import time
import argparse
import onnxruntime
import numpy as np

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', default="images/b.jpeg", type=str, help='path of the input image (a file)')
    parser.add_argument('--output-path', default="./out.png", type=str,
                        help='paht for saving the predicted alpha matte (a file)')
    parser.add_argument('--model-path', default="./onnx/hair.onnx", type=str,
                        help='path of the ONNX model')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.image_path):
        print('Cannot find the input image: {0}'.format(args.image_path))
        exit()
    if not os.path.exists(args.model_path):
        print('Cannot find the ONXX model: {0}'.format(args.model_path))
        exit()

    ref_size = 512

    ##############################################
    #  Main Inference part
    ##############################################

    # read image
    im = cv2.imread(args.image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # normalize values to scale it between -1 to 1
    im = (im - 127.5) / 127.5
    im_h, im_w, im_c = im.shape

    # resize image
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)

    # prepare input shape
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis=0).astype('float32')

    # Initialize session and get prediction
    session = onnxruntime.InferenceSession(args.model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("开始测速>>>")
    start_time = time.time()
    for i in range(100):
        result = session.run([output_name], {input_name: im})
        result=result[0][0]
        print(result.min(),result.max())
        cv2.imshow("hah",result)
        cv2.waitKey(0)
        print(result[0][0].shape)
    print("100次推理onnx耗时:", (time.time() - start_time))

    # refine matte
    print()
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(args.output_path, matte)
