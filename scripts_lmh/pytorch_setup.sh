module load python/3.8.5-fasrc01 cuda/11.1.0-fasrc01 cudnn/8.0.4.30_cuda11.1-fasrc01 && source activate pt1.8_cuda111
module load gcc/7.1.0-fasrc01

export BYTEPS_NCCL_HOME=/n/home03/minghaoli/.conda/pkgs/nccl-2.11.4.1-h97a9cb7_0

export BYTEPS_NUMA_ON=0

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/home03/minghaoli/.conda/pkgs/nccl-2.11.4.1-h97a9cb7_0/lib:/n/home03/minghaoli/byteps/byteps/common/test
LIBRARY_PATH=$LIBRARY_PATH:/n/home03/minghaoli/.conda/pkgs/nccl-2.11.4.1-h97a9cb7_0/lib:/n/home03/minghaoli/byteps/byteps/common/test

source activate pt1.8_cuda111
