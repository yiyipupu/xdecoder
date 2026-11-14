export XD_ROOT=${XD_ROOT:-/home/pupu/insPart/XD}
export XD_STORAGE=${XD_STORAGE:-/media/pupu/PUPU/insPart/XD}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.8}
export HF_HOME="$XD_STORAGE/cache/huggingface"
export TORCH_HOME="$XD_STORAGE/cache/torch"
unset PYTHONPATH

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export XD_PYTHON=${XD_PYTHON:-python}
export PATH="$CUDA_HOME/bin:$PATH"
