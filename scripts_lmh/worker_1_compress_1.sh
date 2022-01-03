export BYTEPS_THREADPOOL_SIZE=16
export NVIDIA_VISIBLE_DEVICES=0,1,2,3
export DMLC_WORKER_ID=1
export DMLC_NUM_WORKER=$2
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=$((2*$2))
export DMLC_PS_ROOT_URI=10.31.179.$1  # the scheduler IP
export DMLC_PS_ROOT_PORT=1234 # the scheduler port
