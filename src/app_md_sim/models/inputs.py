from typing import Optional, Dict, Tuple
from pydantic import BaseModel
import json
from pathlib import Path

class ForceFieldParams(BaseModel):
    charges: Dict[str, float]
    vdw_params: Dict[str, Tuple[float, float]]
    pair_interactions: Optional[dict] = None


class Inputs(BaseModel):
    force_field_params: ForceFieldParams

    # Model parameters
    n_layers: int
    lx: int
    ly: int
    lx_spacing: int
    conc_ZnI2: float
    conc_I: Optional[float] = 0.0
    density_water: Optional[float] = 1.0
    layer_spacing: float
    r_vdw: float
    water_thickness: Optional[float] = 0.0
    buffer_pack: Optional[float] = 0.0

    # Paths
    base_dir: str
    run_files_path: str
    hpc_name: Optional[str] = None
    hpc_tar_path: Optional[str] = None # if None it would skip copying to hpc

    # Execution
    run_on_hpc: bool

    @classmethod
    def from_file(cls, file_path: str | Path) -> "Inputs":
        """Load input parameters from a JSON file and return an Inputs instance."""
        file_path = Path(file_path).expanduser()
        with file_path.open("r") as f:
            data = json.load(f)
        return cls(**data)