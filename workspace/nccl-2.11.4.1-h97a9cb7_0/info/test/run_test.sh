

set -ex



test -f "${PREFIX}/include/nccl.h"
test -f "${PREFIX}/lib/libnccl.so"
test ! -f "${PREFIX}/lib/libnccl_static.a"
exit 0
