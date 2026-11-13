# Common settings, sourced by every config in this folder. Set the three roots before running.
#   DATA_ROOT : datasets (see README.md)
#   CKPT_ROOT : trained models; every config writes to $CKPT_ROOT/<model>/seed<seed>/
#   OUT_ROOT  : evaluation outputs and render caches
export DATA_ROOT=${DATA_ROOT:-/path/to/data}
export CKPT_ROOT=${CKPT_ROOT:-/path/to/checkpoints}
export OUT_ROOT=${OUT_ROOT:-/path/to/outputs}
export PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PY=${PY:-python}
# grodino/waterbirds from the Hugging Face Hub is cached here (Arrow files read by data/waterbirds_data.py)
export HF_DATASETS_CACHE=$DATA_ROOT/hf_cache
# CPU threads and allocator settings used for all GPU runs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
