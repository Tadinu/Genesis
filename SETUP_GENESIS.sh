#mkvirtualenv genesis
#workon genesis
pip install genesis-world
pip install torch torchvision torchaudio
pip uninstall matplotlib
pip install taichi
# or in a CONDA env
# https://pytorch.org/get-started/previous-versions/
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia