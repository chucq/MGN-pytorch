import os
from importlib import import_module
from option import args
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import time

def fliphor(inputs):
    inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W
    return inputs.index_select(3,inv_idx)

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


test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

inputdir = args.input_dir
imglist = os.listdir(inputdir)
imglist.sort()
inputlist = []
for imgpname in imglist:
    imgpath = os.path.join(inputdir,imgpname)
    inputlist.append(test_transform(default_loader(imgpath)))

inputs = torch.stack(inputlist,dim=0)

t1 = time.time()
ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
for i in range(2):
    if i==1:
        inputs = fliphor(inputs)
    input_img = inputs.to(device)
    outputs = model(input_img)
    f = outputs[0].data.cpu()
    ff = ff + f

fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
ff = ff.div(fnorm.expand_as(ff))
t2 = time.time()
print('Time for extract all feature:', t2-t1, 's')
result = np.zeros((len(imglist), len(imglist)))
for i in range(len(imglist)):
    for j in range(len(imglist)):
        result[i][j] = np.dot(ff[i],ff[j])

print('Time for extract all features\' cosine similarity:', time.time()-t2, 's')
print('image index:\n',list(enumerate(imglist)))
print('similarity matrix:\n',result)

