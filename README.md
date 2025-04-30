# Getting Started

This scripts depends on Packmol, hence, you'll need to run on a linux system.

On windows, you can use Ubuntu WSL to get this set and running.

## Clone the repo

```shell
git clone https://github.com/bmotevalli/app-md-sim.git
```

```shell
cd app-md-sim
```

## Environment set up

### Install Poetry

poetry is a python package manager. Install it via following steps:

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

Then, add Poetry to your PATH (if not already set): add below to your ~/.bashrc, ~/.zshrc, or shell profile:

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

Then, run poetry install to set up python's virtual environment.

```shell
poetry install
```

```shell
source .venv/bin/activate
```

### ✅ Packmol - Linux

As mentioned earlier, this package depends on Packmol. Use below commands to install
packmol on your ubuntu system.

```shell
sudo apt update
```

```shell
sudo apt install gfortran make git
```

```shell
git clone https://github.com/m3g/packmol.git
```

```shell
cd packmol
```

```shell
make
```

```shell
sudo cp packmol /usr/local/bin/
```

**verify installation:**

```shell
packmol
```

### HPC Auth (Optional)

With this script, you can generate your model and directly run it on a HPC and sync things back when required.
This is an optional thing, you can always generate the models locally and then ship them to your HPC. If you
aim to automate runs, then you would need to set up HPC access to your local machine.

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

# Usage

After installation and activating the virtual environment you can run following commands:

command: `md-sim --help`

command: `md-sim run examples/param_1.json`

where, examples/param_1.json is the config file containing all required inputs.

command: `md-sim sync examples/param_1.json`

where, examples/param_1.json is the same input file used to generate the models and submit runs.
