from pathlib import Path

import torch

# Committed outputs: measured data in results/, figures drawn from it in plots/.
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BYTES_PER_ELEMENT: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}

# H100 SXM HBM3 peak is 3.35 TB/s.
THEORETICAL_BANDWIDTH_GBPS = 3350
