import typer
from app_md_sim.services.gen_models_packmol import run_from_config
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
def test():
    """
    Run a simulation using the parameters in the given JSON config file.
    """
    print("This is just a test.")


if __name__ == "__main__":
    app()