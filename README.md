# Getting Started

This scripts depends on Packmol, hence, you'll need to run on a linux system.

On windows, you can use Ubuntu WSL to get this set and running.

## Clone the repo

## Environment set up

### HPC Auth

- create ssh keys
- add ssh keys to hpc
- add ssh keys to your local shell
- add following to your ~/.ssh/config

```shell
Host <give it a name, e.g. hpc_example>
    HostName <your hpc address>
    User <your-username>
    IdentityFile ~/.ssh/id_rsa_hpc
```

You can then login | access hpc: `ssh hpc_example`

### ❌ Packmol - Windows (depreciated in current version)

To set up packmol in Windows, you would need julia.

- install julia (see https://julialang.org/downloads/platform/)

```shell
cd /tmp
wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.4-linux-x86_64.tar.gz
tar -xvzf julia-1.11.4-linux-x86_64.tar.gz
sudo mv julia-1.11.4
sudo ln -s /opt/julia-1.11.4/bin/julia /usr/local/bin/julia
```

- validate installation: `julia --version`
- install packmol on julia:

```shell
julia
```

Then, in julia

```julia
import Pkg
Pkg.add("Packmol")
```

#### ❌ Integrating julia with python - Windows (depreciated in current version)

Back in python you may need to do following:

- Activate your virtual env
- enter python's shell: `pyhon`
- Run following commands:

```python
>>> import julia
>>> julia.install()
```

### ✅ Packmol - Linux

```shell
sudo apt update
sudo apt install gfortran make git

git clone https://github.com/m3g/packmol.git
cd packmol

make

sudo cp packmol /usr/local/bin/
```

**verify installation:**

```shell
packmol
```

# Set up Python package

## Install Poetry

poetry is a python package manager. Install it via following commands:

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

Then, add Poetry to your PATH (if not already set): add this to your ~/.bashrc, ~/.zshrc, or shell profile:

```
export PATH="$HOME/.local/bin:$PATH"
```

Then reload your shell:

```
source ~/.bashrc  # or ~/.zshrc
```

**Verify installation:**

```
poetry --version
```

## Set up project:

```shell

git clone https://github.com/bmotevalli/app-md-sim.git

cd app-md-sim

poetry install

source .venv/bin/activate
```

Then, you should be able to execute below commands:

command: `md-sim --help`

command: `md-sim run examples/param_1.json`

where, examples/param_1.json is the config file containing all required inputs.

command: `md-sim sync examples/param_1.json`

where, examples/param_1.json is the same input file used to generate the models and submit runs.
