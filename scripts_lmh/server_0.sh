export DMLC_NUM_WORKER=2
export DMLC_ROLE=server
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.31.179.$1  # the scheduler IP
export DMLC_PS_ROOT_PORT=1234  # the scheduler port
export BYTEPS_SERVER_ENGINE_THREAD=10