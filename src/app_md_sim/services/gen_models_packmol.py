"""
@TODO:
    Once debugged and worked fine, refactor this script into several components:
    - proposed strucutre:
        |- services
            |- gen_models
                |- graphene.py
                |- water.py
                |- electorlyte_sys.py
                |- ...
            |- hpc
"""

from typing import List
import numpy as np
from ase import Atoms, Atom
from ase.io import write as write_ase
import os
from julia import Packmol
from collections import defaultdict
from ase.io import read
import re
from pathlib import Path
import shutil
import subprocess
import time
from collections import defaultdict
from scipy.spatial import cKDTree

from app_md_sim.models.inputs import Inputs, ForceFieldParams, FunctionalGroup, FuncGroupType


def create_graphene_system(
        n_layers, 
        lx, 
        ly, 
        lx_spacing, 
        layer_spacing=3.35, 
        carbon_charge=0.0,
        displace_x=0.0,
        func_groups: List[FunctionalGroup] = []
    ):
    """
    Create graphene system with custom dimensions and assign charges to carbon atoms if required.

    Parameters:
    -----------
    n_layers : int
        Number of graphene layers.
    lx : float
        Desired length of graphene sheet in x-direction (Angstroms).
    ly : float
        Desired width of graphene sheet in y-direction (Angstroms).
    lx_spacing : float
        spacing around graphene sheets in x-direction.
    layer_spacing : float
        Interlayer spacing in Angstroms (default: 3.35 Å).
    carbon_charge : float
        Charge to assign to each carbon atom (default: 0.0 for neutral graphene).
    displace_x : float
        It displaces odd layers to create channels of graphene layers

    Returns:
    --------
    atoms : ASE Atoms object
        Graphene structure with assigned charges.
    """
    # Graphene lattice parameters
    a = 2.46  # Lattice constant in Angstroms
    
    # Calculate number of unit cells needed
    nx = int(np.ceil(lx / a))  # Number of unit cells in x
    ny = int(np.ceil(ly / (a * np.sqrt(3)/2)))  # Number of unit cells in y
    
    # Create multilayer graphene structure
    atoms = create_multilayer_graphene(n_layers, nx, ny, layer_spacing)

    # This is to create channels of graphene when displace_x is flagged
    if displace_x != 0.0:
        positions = atoms.get_positions()
        for layer in range(n_layers):
            if layer % 2 == 1:
                # Get z coordinate of this layer
                z_val = layer * layer_spacing
                # Find atom indices in this layer (within small tolerance)
                toler = 0.4
                indices = np.where((positions[:, 2] >= z_val - toler) & (positions[:, 2] < z_val + toler))[0]
                positions[indices, 0] += displace_x

        atoms.set_positions(positions)

    if abs(carbon_charge) > 0.0001:
        atoms_2 = atoms.copy() # create_multilayer_graphene(n_layers, nx, ny, layer_spacing)
        # SET CHARGES:
        charges_p = np.full(len(atoms), carbon_charge)  # Set charge for all atoms
        atoms.set_initial_charges(charges_p)

        charges_n = np.full(len(atoms_2), -1 * carbon_charge)  # Set charge for all atoms
        atoms_2.set_initial_charges(charges_n)

        # DISPLACE second set of graphen layers
        positions = atoms_2.get_positions()
        max_atoms_2_x = np.max(positions[:, 0])
        positions[:,0] += max_atoms_2_x + 2 * lx_spacing
        atoms_2.set_positions(positions)
        atoms += atoms_2

    if func_groups:
        atoms = apply_functional_groups(atoms, n_layers, func_groups, layer_spacing)

    # Shifting atoms by lx_spacing to create space from origin 0.0
    positions = atoms.get_positions()
    positions[:,0] += lx_spacing
    positions[:,2] += layer_spacing / 2 # Centering the layers
    atoms.set_positions(positions)
    max_x = np.max(positions[:, 0])
    
    # Get current cell dimensions
    current_cell = atoms.get_cell()    
    # Calculate the new cell dimensions
    new_cell = current_cell.copy()
    new_cell[0,0] = max_x + lx_spacing  # Set new x length
    
    # # Calculate offset to center the graphene sheet in the cell
    # x_offset = (lx_spacing - current_cell[0,0]) / 2
    
    # # Update atomic positions to center the sheet
    # positions = atoms.get_positions()
    # positions[:,0] += x_offset
    # atoms.set_positions(positions)
    
    # Update cell and periodic boundary conditions
    atoms.set_cell(new_cell)
    atoms.set_pbc([True, True, True])  # Periodic only in y-direction
    
    return atoms


def create_multilayer_graphene(n_layers, nx=4, ny=4, layer_spacing=3.35):
    """
    Create multi-layer graphene with proper honeycomb structure
    All atoms contained within cell boundaries
    
    Parameters:
    -----------
    n_layers : int
        Number of graphene layers
    nx, ny : int
        Number of unit cells in x and y directions
    layer_spacing : float
        Interlayer spacing in Angstroms (default: 3.35Å)
    
    Returns:
    --------
    atoms : ASE Atoms object
        Graphene structure with honeycomb arrangement
    """
    # Graphene lattice parameters
    a = 2.46  # Lattice constant in Angstroms
    c_c = 1.42  # C-C bond length in Angstroms
    
    # Define primitive vectors (60-degree angle between them)
    a1 = a * np.array([1.0, 0.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3)/2, 0.0])
    
    # Define basis vectors for the two atoms in primitive cell
    basis1 = np.array([0.0, 0.0, 0.0])
    basis2 = np.array([a/2, a/(2*np.sqrt(3)), 0.0])
    
    positions = []
    
    # Calculate cell dimensions first
    cell_x = nx * a
    cell_y = ny * a * np.sqrt(3)/2
    cell_z = n_layers * layer_spacing
    
    # Create each layer
    for layer in range(n_layers):
        z = layer * layer_spacing
        
        # AB stacking shift for odd-numbered layers
        shift = np.array([a/2, a/(2*np.sqrt(3)), 0.0]) if layer % 2 else np.array([0.0, 0.0, 0.0])
        
        # Create atoms in the layer
        for i in range(nx):
            for j in range(ny):
                # Position of the unit cell
                cell_pos = i * a1 + j * a2
                
                # Add the two basis atoms
                for basis in [basis1, basis2]:
                    pos = cell_pos + basis + shift
                    
                    # Ensure position is within cell boundaries
                    x = pos[0]
                    y = pos[1]
                    
                    # Shift atoms to be fully within the cell
                    if x >= cell_x:
                        x -= cell_x
                    if y >= cell_y:
                        y -= cell_y
                    
                    positions.append(np.array([x, y, z]))
    
    # Create cell dimensions
    cell = [cell_x, cell_y, cell_z]
    
    # Create ASE Atoms object
    atoms = Atoms('C' * len(positions),
                 positions=positions,
                 cell=cell,
                 pbc=[True, True, True])
    
    # Sort atoms by z-coordinate then x-coordinate for cleaner visualization
    positions = atoms.get_positions()
    sort_idx = np.lexsort((positions[:,0], positions[:,2]))
    atoms = atoms[sort_idx]
    
    return atoms


def apply_functional_groups(
    atoms,
    n_layers: int,
    func_groups: List[FunctionalGroup],
    layer_spacing=3.35,
    z_tolerance=0.4,
    seed=None
):
    """
    Apply functional groups layer-by-layer with consistent ratio per layer,
    ensuring no overlapping functionalization and randomized placement.

    Parameters:
    -----------
    atoms : ASE Atoms
    n_layers : int
    func_groups : List[FunctionalGroup]
    layer_spacing : float
    z_tolerance : float
    seed : int

    Returns:
    --------
    ASE Atoms object (functionalized)
    """
    if seed is not None:
        np.random.seed(seed)

    positions = np.array(atoms.get_positions())
    symbols = atoms.get_chemical_symbols()
    initial_charges = atoms.get_initial_charges()
    if np.all(initial_charges == 0.0):
        initial_charges = np.zeros(len(atoms))

    new_atoms = atoms.copy()
    charges = list(initial_charges)
    used_carbon_indices = set()  # Track all used carbon atoms across layers

    for layer in range(n_layers):
        z_center = layer * layer_spacing
        indices = np.where((positions[:, 2] >= z_center - z_tolerance) &
                           (positions[:, 2] < z_center + z_tolerance))[0]
        layer_carbons = [i for i in indices if symbols[i] == 'C' and i not in used_carbon_indices]
        layer_positions = positions[layer_carbons]

        for group in func_groups:
            ratio = group.ratio
            gtype = group.name
            group_charges = group.charges or {}

            if gtype == FuncGroupType.OH:
                n_attach = int(len(layer_carbons) * ratio)
                if n_attach == 0:
                    continue

                selected = np.random.choice(layer_carbons, size=n_attach, replace=False)

                for idx in selected:
                    c_pos = positions[idx]

                    # Place O atom above carbon
                    o_pos = c_pos + np.array([0, 0, 1.43])

                    # Place H atom at ~104.5° angle in xz plane
                    angle_deg = 104.5
                    angle_rad = np.radians(angle_deg)
                    oh_bond_length = 0.97
                    dx = oh_bond_length * np.sin(angle_rad)
                    dz = oh_bond_length * np.cos(angle_rad)
                    h_pos = o_pos + np.array([dx, 0.0, -dz])  # tilt H back from O

                    new_atoms += Atom('O', position=o_pos)
                    new_atoms += Atom('H', position=h_pos)
                    charges.append(group_charges.get('O', 0.0))
                    charges.append(group_charges.get('H', 0.0))

                    if 'C' in group_charges:
                        charges[idx] = group_charges['C']

                    used_carbon_indices.add(idx)

            elif gtype == FuncGroupType.O:
                n_target_pairs = int(len(layer_carbons) * ratio / 2)
                if n_target_pairs == 0:
                    continue

                pairs = []
                shuffled_indices = np.random.permutation(len(layer_positions))
                for i_local in shuffled_indices:
                    i_global = layer_carbons[i_local]
                    if i_global in used_carbon_indices:
                        continue
                    for j_local in shuffled_indices:
                        if i_local == j_local:
                            continue
                        j_global = layer_carbons[j_local]
                        if j_global in used_carbon_indices:
                            continue
                        dist = np.linalg.norm(layer_positions[i_local] - layer_positions[j_local])
                        if 1.3 < dist < 1.6:
                            pairs.append((i_global, j_global))
                            used_carbon_indices.update([i_global, j_global])
                            break  # Only one partner per i
                    if len(pairs) >= n_target_pairs:
                        break

                for i1, i2 in pairs:
                    pos1 = positions[i1]
                    pos2 = positions[i2]
                    o_pos = 0.5 * (pos1 + pos2) + np.array([0, 0, 1.43])
                    new_atoms += Atom('O', position=o_pos)
                    charges.append(group_charges.get('O', 0.0))

                    if 'C' in group_charges:
                        charges[i1] = group_charges['C']
                        charges[i2] = group_charges['C']

            else:
                raise ValueError(f"Unsupported functional group type: {gtype}")

    new_atoms.set_initial_charges(charges)
    return new_atoms





def add_cryst1_to_pdb(pdb_file, lx_cell, ly, total_height):
    """
    Add CRYST1 record to define the periodic cell in a PDB file.
    
    Parameters:
    -----------
    pdb_file : str
        Path to the PDB file.
    lx_cell : float
        Box dimension in the x-direction (Å).
    ly : float
        Box dimension in the y-direction (Å).
    total_height : float
        Box dimension in the z-direction (Å).
    """
    cryst1_line = f"CRYST1 {lx_cell:8.3f} {ly:8.3f} {total_height:8.3f}  90.00  90.00  90.00 P 1           1\n"
    with open(pdb_file, "r") as f:
        lines = f.readlines()
    
    # Insert CRYST1 as the first line of the file
    if lines[0].startswith("HEADER"):
        lines.insert(1, cryst1_line)
    else:
        lines.insert(0, cryst1_line)
    
    with open(pdb_file, "w") as f:
        f.writelines(lines)



def run_packmol(input_file, working_dir=None):
    cmd = ["packmol"]
    curr_dir = os.getcwd()

    try:
        if working_dir:
            os.chdir(working_dir)
            input_path = input_file
        else:
            input_path = input_file

        with open(input_path, "r") as inp:
            print(f"🧪 Running Packmol in {os.getcwd()} with input: {input_path}")
            subprocess.run(cmd, stdin=inp, check=True)
            print("✅ Packmol completed successfully.")
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Packmol failed with return code {e.returncode}")
    
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    
    finally:
        os.chdir(curr_dir)


def create_packmol_inp_graphene_multilayer_electrolyte(
        base_dir: Path,
        n_layers: int,
        lx: float,
        ly: float,
        lx_spacing: float,
        layer_spacing: float = 3.35,
        displace_x: float = 0.0,
        conc_ZnI2: float = 0.0,
        conc_I: float = 0.0,
        density_water: float = 1.0,
        r_vdw: float = 2.0,
        water_thickness = 0.0,
        carbon_charge = 0.0,
        buffer_pack = 0.0,
        func_groups: List[FunctionalGroup] = []
    ):
    """
    Creates the .pdb base files for multi-layered graphene and
    the required molecules / ions.
    
    Parameters:
    -----------
    base_dir      : The base working directory
    n_layers      : number of layers.
    lx            : length of graphene layer
    ly            : width of graphene layer
    lx_spacing    : spacing in x-direction around graphene
    layer_spacing :
    conc_ZnI2     : amount of concentration for ZnI2
    conc_I        : amount of concentration for pure I (this might be extra)
    density_water : density of water
    r_vdw         : vdw radius used to calculate volume
    charge        : if defined, it will generate two multilayered system
                    one with postive charge and the other with negative charge.

    water_thickness : if defined adds a column of water at the top of the
                      graphene layer, isolating the multilayer
    """
    # ===================================
    # CREATE WORKING DIRECTORY
    # ===================================
    folder = f"graphene_n{n_layers}_s{layer_spacing}_lx{lx}_ly{ly}_lxSpace{lx_spacing}_wt{water_thickness}_ZnI2_{conc_ZnI2}_chg_{carbon_charge}"
    working_dir = os.path.join(base_dir, folder)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    
    # ===================================
    # CREATE PDB FILES
    # ===================================
    graphene = create_graphene_system(n_layers, lx, ly, lx_spacing, layer_spacing, carbon_charge, displace_x, func_groups)

    lx_cell = graphene.get_cell()[0,0]

    print(f"The cell size is {graphene.get_cell()}")

    # Original positions
    water_pos = np.array([
        [0.00, 0.00, 0.00],    # O
        [0.96, 0.00, 0.00],    # H
        [-0.32, 0.93, 0.00]    # H
    ])

    # Recenter around center of geometry (simple average of positions)
    center = water_pos.mean(axis=0)
    water_pos -= center

    water = Atoms(['O', 'H', 'H'],
              positions=water_pos)
    zn = Atoms('Zn', positions=[[0, 0, 0]])
    iodine = Atoms('I', positions=[[0, 0, 0]])


    write_ase(os.path.join(working_dir, "graphene.pdb"), graphene)
#     with open(os.path.join(working_dir,"water.pdb"), 'w') as f:
#         f.write("""\
# ATOM      1  O   HOH     1      12.345  23.456  34.567   1.00  0.00
# ATOM      2  H1  HOH     1      12.500  23.600  34.700   1.00  0.00
# ATOM      3  H2  HOH     1      12.200  23.300  34.400   1.00  0.00
# END
# """)

    write_ase(os.path.join(working_dir,"water.pdb"), water)
    write_ase(os.path.join(working_dir,"Zn.pdb"), zn)
    write_ase(os.path.join(working_dir,"I.pdb"), iodine)

    # write_ase(os.path.join(working_dir, "graphene.xyz"), graphene)
    # write_ase(os.path.join(working_dir,"water.xyz"), water)
    # write_ase(os.path.join(working_dir,"Zn.xyz"), zn)
    # write_ase(os.path.join(working_dir,"I.xyz"), iodine)



    graphene_pos = graphene.get_positions()
    symbols = graphene.get_chemical_symbols()
    positions = np.array(graphene.get_positions())
    carbon_positions = positions[np.array([s == 'C' for s in symbols])]
    graphene_z_positions = np.unique(np.round(carbon_positions[:,2], decimals=2))
    min_z = np.min(graphene_z_positions)
    max_z = np.max(graphene_z_positions)
    x_start = np.min(graphene_pos[:,0])
    x_end = np.max(graphene_pos[:,0])

    water_exclusion_zones = []
    for z_graphene in graphene_z_positions:
        zone = {
            'x_min': x_start - r_vdw,
            'x_max': x_end + r_vdw,
            'y_min': 0,
            'y_max': ly,
            'z_min': z_graphene - r_vdw,
            'z_max': z_graphene + r_vdw
        }
        water_exclusion_zones.append(zone)
    
    # Define single exclusion volume for ions (spanning all layers)
    ion_exclusion_zone = {
        'x_min': x_start - r_vdw,
        'x_max': x_end + r_vdw,
        'y_min': 0,
        'y_max': ly,
        'z_min': min_z - r_vdw,
        'z_max': max_z + layer_spacing - r_vdw
    }
    
    # Calculate total system height including water thickness
    total_height = graphene.cell[2][2]  + water_thickness # + 4 * r_offset

    if water_thickness == 0.0:
        ion_exclusion_zone['z_min'] = - r_vdw
        ion_exclusion_zone['z_max'] = total_height + r_vdw
    
    
    # Calculate available volume (excluding exclusion zones)
    def calculate_available_volume():
        total_volume = lx_cell * ly * (total_height)
        water_excluded = sum((zone['x_max'] - zone['x_min']) * 
                           (zone['y_max'] - zone['y_min']) * 
                           (zone['z_max'] - zone['z_min']) 
                           for zone in water_exclusion_zones)
        return total_volume - water_excluded
    
    available_volume_A3 = calculate_available_volume()
    available_volume_L = available_volume_A3 * 1e-27  # Convert Å³ to L
    
    # Calculate number of molecules/ions needed
    N_A = 6.022e23  # Avogadro's number
    n_ZnI2 = int(conc_ZnI2 * available_volume_L * N_A)
    n_I = int(conc_I * available_volume_L * N_A)
    mass_water = density_water * available_volume_L * 1000
    n_water = int((mass_water / 18.015) * N_A)

    print(f"Number of required water molecules: {n_water}")
    print(f"Number of required ZnI2: {n_ZnI2}")
    print(f"Number of required I: {n_I}")
    print(f"Total height: {total_height}")



    packmol_str = f"""
tolerance 2.0
filetype pdb
output graphene_electrolyte.pdb

# Fixed graphene layers

structure graphene.pdb
  number 1
  fixed 0.0 0.0 0.0 0.0 0.0 0.0
end structure


# Zn ions
structure Zn.pdb
  number {n_ZnI2}
  inside box {buffer_pack} {buffer_pack} {buffer_pack}  {lx_cell - buffer_pack} {ly - buffer_pack} {total_height - buffer_pack}
end structure

# I ions
structure I.pdb
  number {n_ZnI2 * 2 + n_I}
  inside box {buffer_pack} {buffer_pack} {buffer_pack}  {lx_cell - buffer_pack} {ly - buffer_pack} {total_height - buffer_pack}
end structure

# Water molecules
structure water.pdb
  number {n_water}
  inside box {buffer_pack} {buffer_pack} {buffer_pack}  {lx_cell - buffer_pack} {ly - buffer_pack} {total_height - buffer_pack}
end structure
"""
    with open(os.path.join(working_dir, "graphene_electrolyte_packmol.inp"), 'w') as f:
        f.write(packmol_str)

    # Packmol.run_packmol(os.path.join(working_dir, "graphene_electrolyte_packmol.inp"))

    run_packmol("graphene_electrolyte_packmol.inp", working_dir)

    # Adding a delay to ensure packmol has written the file.
    time.sleep(1)
    add_cryst1_to_pdb(
        pdb_file=os.path.join(working_dir, "graphene_electrolyte.pdb"),
        lx_cell=lx_cell,
        ly=ly,
        total_height=total_height
    )

    return folder


def parse_pdb_molecule_ids(pdb_file):
    """
    Parses molecule IDs from a PDB file, handling different formatting cases.

    Parameters:
    -----------
    pdb_file : str
        Path to the PDB file.

    Returns:
    --------
    molecule_ids : list
        List of molecule IDs for each atom in the order they appear in the PDB file.
    """
    molecule_ids = []
    next_id = 1
    prev_mol_id = 1

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = re.split(r'\s+', line.strip())  # Split by whitespace
                
                if len(parts) >= 6:
                    # Handle cases where molecule ID is a clean number
                    try:
                        molecule_id = int(parts[4])
                    except ValueError:
                        if next_id == 1:
                            next_id = 2
                        if len(parts) > 11:
                            try:
                                # Handle cases like "C 238" (strip non-numeric characters)
                                molecule_id = int(parts[5])

                            except ValueError:
                                # Handle cases like "C1238" (strip non-numeric characters)
                                molecule_id_str = re.sub(r'[^0-9]', '', parts[4])  # Keep only digits
                                if molecule_id_str:
                                    molecule_id = int(molecule_id_str)
                                else:
                                    raise ValueError(f"Could not parse molecule ID from line: {line.strip()}")
                            
                            if molecule_id != prev_mol_id:
                                next_id += 1
                                prev_mol_id = molecule_id
                    

                    molecule_ids.append(next_id)

    return molecule_ids


def write_lammps_data_from_pdb(pdb_file, charges, vdw_params, func_groups=[], filename="graphene_electrolyte.data"):
    """
    Reads a PDB file and writes a LAMMPS data file including custom charges, vdW parameters, and molecular topology.
    Automatically assigns charges to functional group atoms and their bonded carbon atoms.
    Handles cases where no functional groups are present.
    """
    atoms = read(pdb_file)
    molecule_ids = parse_pdb_molecule_ids(pdb_file)

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)
    unique_symbols = sorted(set(symbols))
    atom_types = {symbol: i + 1 for i, symbol in enumerate(unique_symbols)}

    cell = atoms.get_cell()
    xlo, xhi = 0.0, cell[0, 0]
    ylo, yhi = 0.0, cell[1, 1]
    zlo, zhi = 0.0, cell[2, 2]

    molecule_dict = defaultdict(list)
    for i, mol_id in enumerate(molecule_ids):
        molecule_dict[mol_id].append(i + 1)

    n_bonds, n_angles = 0, 0
    for mol_atoms in molecule_dict.values():
        if len(mol_atoms) == 3:
            n_bonds += 2
            n_angles += 1

    n_carbon = len([i for i in molecule_dict[1] if symbols[i - 1] == 'C'])

    # === Functional group processing (optional) ===
    fg_charge_map = {fg.name.value: fg.charges for fg in func_groups}
    c_fg_contributions = defaultdict(float)

    if func_groups:
        mol1_indices = [i for i, mid in enumerate(molecule_ids) if mid == 1]
        c_indices = [i for i in mol1_indices if symbols[i] == 'C']
        o_indices = [i for i in mol1_indices if symbols[i] == 'O']
        h_indices = [i for i in mol1_indices if symbols[i] == 'H']

        c_pos = np.array([positions[i] for i in c_indices])
        o_pos = np.array([positions[i] for i in o_indices])
        h_pos = np.array([positions[i] for i in h_indices])

        tree_C = cKDTree(c_pos)
        tree_H = cKDTree(h_pos)

        for o_idx, o_xyz in zip(o_indices, o_pos):
            c_near = tree_C.query_ball_point(o_xyz, 1.6)
            h_near = tree_H.query_ball_point(o_xyz, 1.2)

            if len(h_near) == 1 and len(c_near) == 1:
                c_global = c_indices[c_near[0]]
                c_fg_contributions[c_global] += fg_charge_map.get("OH", {}).get("C", 0.0)
            elif len(h_near) == 0 and len(c_near) == 2:
                c1 = c_indices[c_near[0]]
                c2 = c_indices[c_near[1]]
                c_fg_contributions[c1] += fg_charge_map.get("O", {}).get("C", 0.0)
                c_fg_contributions[c2] += fg_charge_map.get("O", {}).get("C", 0.0)

    # === Write the LAMMPS .data file ===
    with open(filename, "w") as f:
        f.write("LAMMPS data file via ASE export with custom charges and vdW parameters\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_bonds} bonds\n")
        f.write(f"{n_angles} angles\n\n")
        f.write(f"{len(unique_symbols)} atom types\n")
        f.write("1 bond types\n")
        f.write("1 angle types\n\n")

        f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")

        f.write("Masses\n\n")
        for symbol, atom_type in atom_types.items():
            mass = atoms[[s == symbol for s in symbols]][0].mass
            f.write(f"{atom_type} {mass:.4f} # {symbol}\n")
        f.write("\n")

        f.write("Pair Coeffs\n\n")
        for symbol, atom_type in atom_types.items():
            epsilon, sigma = vdw_params.get(symbol, (0.0, 0.0))
            f.write(f"{atom_type} {epsilon:.4f} {sigma:.4f} # {symbol}\n")
        f.write("\n")

        f.write("Bond Coeffs\n\n")
        f.write("1 450.0 0.96 # O-H bond (harmonic)\n\n")
        f.write("Angle Coeffs\n\n")
        f.write("1 55.0 104.5 # H-O-H angle (harmonic)\n\n")

        f.write("Atoms\n\n")
        count_c = 1
        for i, (pos, symbol, mol_id) in enumerate(zip(positions, symbols, molecule_ids)):
            atom_type = atom_types[symbol]

            if mol_id == 1:
                if func_groups and symbol == "H":
                    charge = fg_charge_map.get("OH", {}).get("H", charges.get("H", 0.0))
                elif func_groups and symbol == "O":
                    is_OH = False
                    if i > 0 and symbols[i - 1] == "H" and molecule_ids[i - 1] == 1:
                        is_OH = True
                    elif i + 1 < len(symbols) and symbols[i + 1] == "H" and molecule_ids[i + 1] == 1:
                        is_OH = True
                    if is_OH:
                        charge = fg_charge_map.get("OH", {}).get("O", charges.get("O", 0.0))
                    else:
                        charge = fg_charge_map.get("O", {}).get("O", charges.get("O", 0.0))
                elif symbol == "C":
                    count_c += 1
                    base = charges.get("C", 0.0)
                    if count_c > n_carbon / 2:
                        base *= -1
                    fg = c_fg_contributions.get(i, 0.0)
                    charge = base + fg
                else:
                    charge = charges.get(symbol, 0.0)
            else:
                charge = charges.get(symbol, 0.0)

            f.write(f"{i + 1} {mol_id} {atom_type} {charge:.4f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

        f.write("\nBonds\n\n")
        bond_id = 1
        for mol_atoms in molecule_dict.values():
            if len(mol_atoms) == 3:
                f.write(f"{bond_id} 1 {mol_atoms[0]} {mol_atoms[1]}\n")
                bond_id += 1
                f.write(f"{bond_id} 1 {mol_atoms[0]} {mol_atoms[2]}\n")
                bond_id += 1

        f.write("\nAngles\n\n")
        angle_id = 1
        for mol_atoms in molecule_dict.values():
            if len(mol_atoms) == 3:
                f.write(f"{angle_id} 1 {mol_atoms[1]} {mol_atoms[0]} {mol_atoms[2]}\n")
                angle_id += 1

    print(f"LAMMPS data file written to {filename}")



def copy_run_files(src_dir, dest_dir):
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Loop through all files in the source directory
    for file_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)

        # Copy only files (ignore directories)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)  # `copy2` preserves metadata

    print(f"All files copied from {src_dir} to {dest_dir}")


def copy_files_to_hpc(source_path, hpc_name, dest_path):
    """
    Ensures the destination directory exists on the HPC, then copies files using SCP.

    Parameters:
    -----------
    source_path : str
        Local path of the files to copy.
    hpc_name : str
        SSH alias for the HPC (e.g., "hpc_vigra").
    dest_path : str
        Destination path on the HPC (relative to home directory).
    """
    # SSH command to check and create the destination directory on the HPC
    ssh_command = [
        "ssh", hpc_name, f"mkdir -p ~/{dest_path}"
    ]

    # SCP command to copy files
    # scp_command = [
    #     "scp", "-r", f"{source_path.rstrip('/')}/.", f"{hpc_name}:~/{dest_path}"
    # ]

    # Better copy for contents of a folder
    rsync_command = [
        "rsync", "-av", f"{source_path.rstrip('/')}/", f"{hpc_name}:~/{dest_path}/"
    ]

    try:
        # Ensure destination directory exists
        print(f"Ensuring {dest_path} exists on {hpc_name}...")
        subprocess.run(ssh_command, check=True)

        # Copy files
        print(f"Copying {source_path} to {hpc_name}:~/{dest_path}...")
        # subprocess.run(scp_command, check=True)
        subprocess.run(rsync_command, check=True)

        print(f"✅ Successfully copied {source_path} to {hpc_name}:~/{dest_path}")
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Command failed with return code {e.returncode}")
    except FileNotFoundError:
        print("❌ Error: SSH or SCP command not found. Ensure SSH is installed.")


def sync_entire_dir_from_hpc(hpc_name, base_remote_dir, base_local_dir):
    """
    Syncs an entire directory tree from HPC to local using rsync, copying only
    newer or missing files.

    Parameters:
    -----------
    hpc_name : str
        SSH alias for the HPC (e.g., "hpc_vigra").
    base_remote_dir : str
        Base directory on the HPC (relative to home, e.g., "projects/md-sim/sims").
    base_local_dir : str
        Local base directory (e.g., "~/projects/md-sim/sims").
    """
    base_local_dir = Path(base_local_dir).expanduser().resolve()
    base_local_dir.mkdir(parents=True, exist_ok=True)

    remote_path = f"{hpc_name}:~/{base_remote_dir.rstrip('/')}/"
    local_path = f"{str(base_local_dir)}/"

    rsync_command = [
        "rsync", "-avzu",  # a: archive, v: verbose, z: compress, u: only update if newer
        "--progress",
        remote_path,
        local_path
    ]

    print(f"🔄 Syncing from {remote_path} to {local_path}")
    try:
        subprocess.run(rsync_command, check=True)
        print(f"✅ Sync complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ rsync failed: {e}")
    except FileNotFoundError:
        print("❌ Error: rsync not found. Ensure rsync is installed on your system.")



def submit_job_to_hpc(hpc_name, hpc_tar_path):
    """
    Submits a SLURM job to the HPC using sbatch and retrieves the job ID.

    Parameters:
    -----------
    hpc_name : str
        SSH alias for the HPC (e.g., "hpc_vigra").
    hpc_tar_path : str
        Path to the job script directory on the HPC.

    Returns:
    --------
    job_id : str or None
        SLURM job ID if submission was successful, else None.
    """
    # Construct SSH command to submit the job
    print(f"Running job at path: ~/{hpc_tar_path}")
    run_cmd = ["ssh", hpc_name, f"cd ~/{hpc_tar_path} && sbatch run.sh"]

    try:
        # Submit the job and capture the output
        print("Submitting the job to the HPC...")
        result = subprocess.run(run_cmd, check=True, text=True, capture_output=True)

        # Extract Job ID from the output (format: "Submitted batch job 12345")
        output = result.stdout.strip()
        print(f"🔹 SLURM Output: {output}")

        if "Submitted batch job" in output:
            job_id = output.split()[-1]  # Extract the last word (job ID)
            print(f"✅ Job submitted successfully! Job ID: {job_id}")
            return job_id
        else:
            print("❌ Error: Unable to extract job ID.")
            return None

    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Command failed with return code {e.returncode}")
        return None



def submit_run(
    base_dir: Path,
    force_field_params: ForceFieldParams,
    n_layers: int,
    lx: float,
    ly: float,
    lx_spacing: float,
    layer_spacing: float = 3.35,
    displace_x: float = 0.0,
    conc_ZnI2: float = 0.0,
    conc_I: float = 0.0,
    density_water: float = 1.0,
    r_vdw: float = 2.0,
    water_thickness = 0.0,
    func_groups: List[FunctionalGroup] = [],
    hpc_name: str = "hpc_vigra",
    run_files_path: Path = None,
    hpc_tar_path: str = None,
    run_on_hpc: bool = False,
    buffer_pack: float = 0.0
):
    """
    The function does followings:

    1. Generates multi-layered graphene (zero charge or charged cases)
    2. Stores the graphene, water molecules, Ions in .pdb form
    3. Calculates volume and number of required water molecules and ions
    4. Generates the .inp file for packmol
    5. Runs packmol with julia to generate the electrolyte in .pdb form
    6. Runs scripts to convert the .pdb file to .data file for lammps
        Force-field params should be inputted
    7. If flagged, it will submit a lammps job to hpc
    8. Potentially, add a monitoring job to monitor the hpc jobs and report
    
    Parameters:
    -----------
    base_dir           : The base working directory
    force_field_params : Includes charges and vdw params.
    n_layers           : number of layers.
    lx                 : length of graphene layer
    ly                 : width of graphene layer
    lx_spacing         : spacing in x-direction around graphene
    layer_spacing      :
    conc_ZnI2          : amount of concentration for ZnI2
    conc_I             : amount of concentration for pure I (this might be extra)
    density_water      : density of water
    r_vdw              : vdw radius used to calculate volume
    water_thickness    : if defined adds a column of water at the top of the
                         graphene layer, isolating the multilayer
    """

    carbon_charge = force_field_params.charges.get("C", 0.0)
    folder = create_packmol_inp_graphene_multilayer_electrolyte(
        base_dir=base_dir,
        n_layers=n_layers,
        lx = lx,
        ly = ly,
        lx_spacing = lx_spacing,
        layer_spacing = layer_spacing,
        displace_x = displace_x,
        conc_ZnI2 = conc_ZnI2,
        conc_I = conc_I,
        density_water = density_water,
        r_vdw = r_vdw,
        water_thickness = water_thickness,
        carbon_charge = carbon_charge,
        buffer_pack=buffer_pack,
        func_groups=func_groups
    )

    # run_pack_mol(base_dir=base_dir, folder=folder)

    if force_field_params is None:
        raise ValueError(f"Force field params are not defined. Not being able to proceed further and generate lammps data.")

    force_field_params.charges["C"] = carbon_charge

    write_lammps_data_from_pdb(
        pdb_file  = os.path.join(base_dir, folder, 'graphene_electrolyte.pdb'),
        charges = force_field_params.charges,
        vdw_params = force_field_params.vdw_params,
        func_groups= func_groups,
        filename=os.path.join(base_dir, folder, "graphene_electrolyte.data")
    )

    if hpc_tar_path is None or run_files_path is None:
        print("Only creating model. Skip copying on HPC.")
        return
    
    hpc_tar_path = os.path.join(hpc_tar_path, folder)
    
    print(f"Copying models and run files to hpc. {hpc_tar_path}")
    # COPY MODEL FILES TO HPC
    copy_files_to_hpc(
        source_path=os.path.join(base_dir, folder),
        hpc_name=hpc_name,
        dest_path=hpc_tar_path
    )

    # COPY RUN FILES TO HPC
    copy_files_to_hpc(
        source_path=run_files_path,
        hpc_name=hpc_name,
        dest_path=hpc_tar_path
    )

    if run_on_hpc:
        print("Submitting the job to HPC!")
        job_id = submit_job_to_hpc(hpc_name, hpc_tar_path)
        if job_id:
            print(f"🎯 Job ID received: {job_id}")
        else:
            print("❌ Job submission failed.")



def run_from_config(config: Inputs):
    submit_run(
        base_dir=config.base_dir,
        force_field_params=config.force_field_params,
        n_layers=config.n_layers,
        lx=config.lx,
        ly=config.ly,
        lx_spacing=config.lx_spacing,
        conc_ZnI2=config.conc_ZnI2,
        conc_I=config.conc_I,
        density_water=config.density_water,
        layer_spacing=config.layer_spacing,
        displace_x=config.displace_x,
        r_vdw=config.r_vdw,
        water_thickness=config.water_thickness,
        func_groups=config.func_groups,
        buffer_pack=config.buffer_pack,
        hpc_name=config.hpc_name,
        run_files_path=config.run_files_path,
        hpc_tar_path=config.hpc_tar_path,
        run_on_hpc=config.run_on_hpc
    )