module load python/3.8.5-fasrc01 cuda/11.0.3-fasrc01 cudnn/8.0.4.30_cuda11.0-fasrc01

module load gcc/4.9.3-fasrc01

export BYTEPS_NCCL_HOME=/n/home03/minghaoli/.conda/pkgs/nccl-2.4.7.1-h51cf6c1_0
export BYTEPS_NUMA_ON=0

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/home03/minghaoli/.conda/pkgs/nccl-2.4.7.1-h51cf6c1_0/lib
LIBRARY_PATH=$LIBRARY_PATH:/n/home03/minghaoli/.conda/pkgs/nccl-2.4.7.1-h51cf6c1_0/lib

source activate mxnet1.8_cuda11_0
