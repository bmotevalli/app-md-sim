import typer
from app_md_sim.services.gen_models_packmol import run_from_config, sync_entire_dir_from_hpc
from app_md_sim.models.inputs import Inputs

app = typer.Typer()

@app.command()
def run(config_file: str = typer.Argument(..., help="Path to your JSON config file")):
    """
    Run a simulation using the parameters in the given JSON config file.
    """
    inputs = Inputs.from_file(config_file)

    run_from_config(inputs)


@app.command()
def sync(config_file: str = typer.Argument(..., help="Path to your JSON config file")):
    """
    This commands syncs your local drive with HPC by downloading back the latest results.
    """
    inputs = Inputs.from_file(config_file)
    
    sync_entire_dir_from_hpc(
        hpc_name = inputs.hpc_name,
        base_remote_dir = inputs.hpc_tar_path,
        base_local_dir = inputs.base_dir,
    )


if __name__ == "__main__":
    app()