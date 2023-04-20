conda create -n gamutmlp_env python=3.9.7
conda activate gamutmlp_env
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda intall -c conda-forge cudatoolkit-dev=11.6.0
export PATH="~/miniconda3/envs/gamutmlp_env/bin:$PATH"
export LD_LIBRARY_PATH="~/miniconda3/envs/gamutmlp_env/lib64:$LD_LIBRARY_PATH"

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install -r requirements.txt