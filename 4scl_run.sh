TIME=$(date +'%Y_%m_%d_%H_%M')
MMCV_DIR='/nas/k8s/dev/research/yongjin117/PROJECT_nautilus/mmcv'
WORK_DIR='/nas/k8s/dev/research/yongjin117/PROJECT_nautilus/MD_RESEARCH/Monocular-Depth-Estimation-Toolbox'

apt-get update
apt-get install -y git

pip3 uninstall -y mmcv-full

cd ${MMCV_DIR}
git config --global --add safe.directory /nas/k8s/dev/research/yongjin117/PROJECT_nautilus/mmcv
git status
MMCV_WITH_OPS=1 pip3 install -e .

pip3 install mmcv-full==1.6.2

cd ${WORK_DIR}
pip3 install rasterio albumentations timm
pip3 install -e .

apt-get install git
git config --global --add safe.directory /nas/k8s/dev/research/yongjin117/PROJECT_nautilus/MD_RESEARCH/Monocular-Depth-Estimation-Toolbox
CUDA_LAUNCH_BLOCKING=1 python3 tools/train.py configs/depthformer/depthformer_swinl_22k_w7_dsm.py --work-dir w_dirs/depthformer/${TIME} --gpus 1
