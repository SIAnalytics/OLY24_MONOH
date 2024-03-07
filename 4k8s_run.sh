TIME=$(date +'%Y_%m_%d_%H_%M')

pip3 install -e .
python3 tools/train.py configs/depthformer/depthformer_swinl_22k_w7_dsm.py --work-dir w_dirs/depthformer/${TIME} --gpus 1
