#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/1/14 14:00
# @Author   : chenkai
# @File     : inference.py

"""
import os
from importlib import import_module
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import time

from config import args
import distance

def fliphor(inputs):
    inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W
    return inputs.index_select(3,inv_idx)

def initialize():
    device = torch.device('cpu' if args.cpu else 'cuda')
    module = import_module('model.mgn')
    model = module.make_model(args).to(device)
    gmodel = model
    if not args.cpu and args.nGPU > 1:
        model = nn.DataParallel(model, range(args.nGPU))
        gmodel = model.module

    model_path = args.model_path
    gmodel.load_state_dict(torch.load(model_path, {}),strict=False)

    model.eval()
    return model


def infer(model, image_list, isFlip = True):
    device = torch.device('cpu' if args.cpu else 'cuda')
    test_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    inputlist = [test_transform(default_loader(imgpath)) for imgpath in image_list]
    inputs = torch.stack(inputlist,dim=0)
    ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
    num = 2 if isFlip else 1
    for i in range(num):
        if i==1:
            inputs = fliphor(inputs)
        input_img = inputs.to(device)
        outputs = model(input_img)
        f = outputs[0].data.cpu()
        ff = ff + f
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))

    return ff


if __name__ == "__main__":
    t1 = time.time()
    mgn_model = initialize()
    t2 = time.time()
    print('Time for init model: %s s'%(t2-t1))

    image_1_path = '/home/ccq/MGN-pytorch/test/zijing/1.png'
    image_2_path = '/home/ccq/MGN-pytorch/test/zijing/2.png'
    feature_list = infer(mgn_model, [image_1_path,image_2_path])
    print('Time for get reid features: %s s'%(time.time() - t2))

    dis = distance.calculate_distance(feature_list[0], feature_list[1])
    print(f'the distance of feature_1 and feature_2 is {dis}')

