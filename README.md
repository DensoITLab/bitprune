# Bit-Pruning
This is the official repo for ICLR 2023 Paper "Bit-Pruning: A Sparse Multiplication-Less Dot-Product"
Yusuke Sekikawa and Shingo Yashima

[paper](https://openreview.net/pdf?id=YUDiZcZTI8), [openreview](https://openreview.net/forum?id=YUDiZcZTI8)


# Usage

## Set GPU(s) to use
```
export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1
```

## Run train and test
```
python main.py --config 'config/cifar10.yaml' 
python main.py --config 'config/cifar100.yaml' 
python main.py --config 'config/imagenet.yaml' 
```
You can change any config by specifying the config name (e.g., optim.spr_w) followed by the value (e.g., `optim.spr_w 16 model.wgt_bit 8 ...`).
use `run_xxx.sh` for batch execution.


## Plot bit-pruning loss 
Run `vis_cost.ipynb` to plot proximal weight and their loss landscape (Fig. 3, Fig. 11).

## Plot histrram of learned weight 
Run `vis_hist.ipynb` to plot the histgram of learned weight distribution (Fig. 6).

# Installation
## Setup docker image from pytorch/pytorch
```
docker pull pytorch/pytorch
docker run -dit --gpus all -v /username/Project/:/home/src -v /data1/dataset:/home/data --name username --shm-size=64gb pytorch/pytorch
docker exec -e CUDA_VISIBLE_DEVICES='0' -u 0 -it username bash
apt-get  -y update && apt-get -y install libgl1 && apt-get  -y install libglib2.0-0
yes | pip install opencv-python
yes | pip install opencv-contrib-python
yes | pip install einops
yes | pip install kornia
yes | pip install lightning-bolts
yes | pip install pytorch-lightning
yes | pip install fvcore
yes | pip install scipy
conda install -c conda-forge easydict --yes 
conda install -c conda-forge ruamel.yaml --yes 
```


If you find our code or paper useful, please cite the following:
```
@inproceedings{iclr2023bitprune,
  author    = {Yusuke, Sekikawa and Shingo, Yashima},
  title     = {Bit-Pruning: A Sparse Multiplication-Less Dot-Product},
  booktitle={Proceedings of the International Conference on Learning Representations},
  year      = {2023},
  url={https://openreview.net/forum?id=YUDiZcZTI8}
}
```