# Multiple Granularity Network
Reproduction of paper:[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)
Original project illustration can be seen at [here](https://github.com/seathiefwang/MGN-pytorch/blob/master/README.md).


## Install dependencies

Install Basic Python dependencies:
```
pip install -r requirements.txt
```

If you need to install PyTorch manually:
```
pip3 install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
```
The command above is used for `Linux`, `Pip`, `Python 3.6` and `CUDA 10.1`, for other situation you can install PyTorch according to [official tutorial](https://pytorch.org/get-started/locally/).

## Run Test Demo

If on offline environment, you should prepare the backbone weights file.
```
mkdir -p ~/.cache/torch/checkpoints/
wget -P ~/.cache/torch/checkpoints/ https://download.pytorch.org/models/resnet50-19c8e357.pth
```

Run the demo script.
```
python test_demo.py --model_path input/model_9383.pt --input_dir input/images
```
The option `--model_path` indicates the pre-trained model file path, and `input_dir`indicates the directory which contains cropped pedestrian images.  

