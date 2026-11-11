# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["numpy==1.26.4", "scipy==1.14.1", "matplotlib==3.10.1"]
# ///
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.update(PYTHONDONTWRITEBYTECODE="1",MPLBACKEND="Agg")
from runner import main
if __name__ == "__main__":
    main('nearest_neighbour')
