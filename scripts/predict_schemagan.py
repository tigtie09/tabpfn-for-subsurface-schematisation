# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["tensorflow-cpu==2.15.1", "numpy==1.26.4", "h5py==3.16.0"]
# ///
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.update(PYTHONDONTWRITEBYTECODE="1",MPLBACKEND="Agg")
os.environ.update(CUDA_VISIBLE_DEVICES="-1",TF_CPP_MIN_LOG_LEVEL="2",TF_DETERMINISTIC_OPS="1",TF_ENABLE_ONEDNN_OPTS="1",TF_NUM_INTRAOP_THREADS="1",TF_NUM_INTEROP_THREADS="1")
from runner import main
if __name__ == "__main__":
    main('schemagan')
