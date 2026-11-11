# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "tabpfn==8.5.0", "torch==2.11.0+cu128", "numpy==2.4.6",
#   "scipy==1.17.1", "scikit-learn==1.9.0", "pandas==3.0.5", "matplotlib==3.10.1",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.update(PYTHONDONTWRITEBYTECODE="1",MPLBACKEND="Agg")
os.environ.update(TABPFN_DISABLE_TELEMETRY="1",HF_HUB_OFFLINE="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
from runner import main
if __name__ == "__main__":
    main('tabpfn')
