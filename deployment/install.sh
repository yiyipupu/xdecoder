#!/bin/bash
set -euxo pipefail
unset PYTHONPATH
P=/home/pupu/miniconda3/envs/instructpart-xdecoder/bin/python
export CUDA_HOME=/usr/local/cuda-11.8
export MAX_JOBS=2
$P -m pip install 'pip<25' 'setuptools<70' wheel ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
$P -m pip install /media/pupu/PUPU/insPart/SAN/wheels/*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
$P -m pip install --no-build-isolation -r /home/pupu/insPart/XD/deployment/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
$P -m pip install --no-build-isolation 'git+https://github.com/MaureenZOU/detectron2-xyz.git@42121d75e10d9f858f3a91b6a39f5722c02868f0' -i https://pypi.tuna.tsinghua.edu.cn/simple
