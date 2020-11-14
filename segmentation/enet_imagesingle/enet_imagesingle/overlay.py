import os
from pynq_dpu import DpuOverlay
overlay = DpuOverlay("dpu.bit")
os.system("dexplorer -w")
