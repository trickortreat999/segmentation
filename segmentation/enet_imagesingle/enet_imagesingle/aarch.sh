#!/bin/bash

set -e

model=segmentation
overlays=overlays_300M2304
cd ${overlays}
#pwd
aarch64-linux-gnu-gcc -fPIC -shared dpu_${model}_0.elf -o libdpumodel${model}.so
echo "aarch64-linux-gnu-gcc -fPIC -shared dpu_${model}_0.elf -o libdpumodel{$model}.so"
cp libdpumodel${model}.so /usr/lib/
ls -l /usr/lib/libdpu*.so
cd ..
pwd
cp ./${overlays}/* /usr/local/lib/python3.6/dist-packages/pynq_dpu/overlays/
python3 overlay.py

