# Getting Started

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

### Packmol

To set up packmol, you would need julia.

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

### Integrating julia with python

Back in python you may need to do following:

- Activate your virtual env
- enter python's shell: `pyhon`
- Run following commands:

```python
>>> import julia
>>> julia.install()
```
