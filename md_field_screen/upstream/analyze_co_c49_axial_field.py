#!/usr/bin/env python3
"""Axial electric field at the Co--C49 bond centre for one GROMACS snapshot.

Charges are read from the supplied TPR.  The whole snapshot is expressed in a
bond-centred frame in which the Co-->C49 vector is +Z.  Co and C49 coordinates
define the axis and field point; the entire TCC residue is excluded.
"""
import argparse
import sys

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import minimize_vectors

# Field in V/nm from q (in e) times r/r^3 (r in nm).
COULOMB_V_NM = 1.439964547


def atom_name(atom):
    return str(atom.name).strip().upper()


def rotation_to_z(vector):
    """Return a proper rotation mapping nonzero vector to the positive Z axis."""
    unit = vector / np.linalg.norm(vector)
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(unit, z_axis)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(unit, z_axis))
    if sine < 1e-12:
        return np.eye(3) if cosine > 0.0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def cutoff_nm(value):
    """Parse a positive spherical C49-centred cutoff, or no cutoff."""
    if value.strip().lower() == "none":
        return None
    try:
        cutoff = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cutoff must be a positive number in nm, or none") from exc
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise argparse.ArgumentTypeError("cutoff must be a positive number in nm, or none")
    return cutoff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gro", required=True, help="single-frame GRO coordinate file")
    parser.add_argument("--tpr", required=True, help="matching GROMACS TPR; supplies charges")
    parser.add_argument("--bin-width", type=float, default=0.10, help="z-bin width in nm (default: 0.10)")
    parser.add_argument("--cutoff", type=cutoff_nm, default=None,
                        help="spherical C49-centred cutoff in nm, or none (default: none)")
    args = parser.parse_args()
    if args.bin_width <= 0:
        parser.error("--bin-width must be positive")

    universe = mda.Universe(args.tpr, args.gro)
    atoms = universe.atoms
    if not hasattr(atoms, "charges"):
        raise RuntimeError("No charges were read from the TPR.")

    # Identify exactly one TCC residue through its C49 and Co atom names.
    candidates = []
    for residue in universe.residues:
        names = [atom_name(atom) for atom in residue.atoms]
        if "C49" in names and "CO" in names:
            candidates.append(residue)
    if len(candidates) != 1:
        c49_residues = [(res.resid, res.resname, [atom.name for atom in res.atoms])
                        for res in universe.residues if "C49" in [atom_name(a) for a in res.atoms]]
        raise RuntimeError(
            "Expected exactly one residue containing both atom names Co and C49; "
            f"found {len(candidates)}. C49-containing residues: {c49_residues}"
        )
    tcc = candidates[0]
    co = [atom for atom in tcc.atoms if atom_name(atom) == "CO"]
    c49 = [atom for atom in tcc.atoms if atom_name(atom) == "C49"]
    if len(co) != 1 or len(c49) != 1:
        raise RuntimeError("The selected TCC residue has non-unique Co or C49 atoms.")
    co, c49 = co[0], c49[0]

    # MDAnalysis coordinates are Angstrom; use nm.  Build the midpoint using
    # the minimum-image Co-->C49 bond, then obtain each source relative to that
    # midpoint before rotation, appropriate for a periodic snapshot.
    box = universe.dimensions.copy()
    c49_relative_nm = minimize_vectors(c49.position - co.position, box) / 10.0
    bond_nm = float(np.linalg.norm(c49_relative_nm))
    rotation = rotation_to_z(c49_relative_nm)
    bond_center = co.position + 0.5 * c49_relative_nm * 10.0
    relative_nm = minimize_vectors(atoms.positions - bond_center, box) / 10.0
    c49_relative_all_nm = minimize_vectors(atoms.positions - c49.position, box) / 10.0
    aligned_nm = relative_nm @ rotation.T

    # Exclude the complete TCC residue, including Co and C49.
    include = atoms.resindices != tcc.ix
    source_r = aligned_nm[include]
    source_q = atoms.charges[include]
    source_resindices = atoms.resindices[include]
    cutoff_radii = np.linalg.norm(c49_relative_all_nm[include], axis=1)
    radii = np.linalg.norm(source_r, axis=1)
    nonzero = radii > 1e-12
    source_r, source_q, source_resindices, radii = (source_r[nonzero], source_q[nonzero],
                                                     source_resindices[nonzero], radii[nonzero])
    cutoff_radii = cutoff_radii[nonzero]
    if args.cutoff is not None:
        within_cutoff = cutoff_radii <= args.cutoff
        source_r, source_q, source_resindices, radii = (
            source_r[within_cutoff], source_q[within_cutoff],
            source_resindices[within_cutoff], radii[within_cutoff]
        )
    if len(source_q) == 0:
        raise RuntimeError("No non-TCC charge sites remain after applying the cutoff.")

    # r is from the bond midpoint (field point) to source.
    contributions = -COULOMB_V_NM * source_q[:, None] * source_r / radii[:, None]**3
    field = contributions.sum(axis=0)

    print("# Co-C49 axial electric field at the bond centre from one snapshot")
    print(f"# GRO: {args.gro}")
    print(f"# TPR: {args.tpr}")
    print(f"# TCC excluded: resid={tcc.resid} resname={tcc.resname} atoms={len(tcc.atoms)}")
    print(f"# Co-C49 bond length: {bond_nm:.6f} nm; midpoint=(0,0,0), Co/C49 lie on -Z/+Z")
    print(f"# C49-centred spherical cutoff: {'none' if args.cutoff is None else f'{args.cutoff:g} nm'}")
    print(f"# included charge sites: {len(source_q)}; total included charge: {source_q.sum():.8f} e")
    print(f"FIELD_X_V_PER_NM {field[0]:.10f}")
    print(f"FIELD_Y_V_PER_NM {field[1]:.10f}")
    print(f"FIELD_Z_V_PER_NM {field[2]:.10f}")
    print(f"AXIAL_FIELD_BOND_CENTER_V_PER_NM {field[2]:.10f}")
    print("# residue_index resid resname n_sites charge_e field_x_V_per_nm field_y_V_per_nm field_z_V_per_nm")
    for residue_index in np.unique(source_resindices):
        present = source_resindices == residue_index
        residue = universe.residues[residue_index]
        residue_field = contributions[present].sum(axis=0)
        print(f"{residue_index} {residue.resid} {residue.resname} {present.sum()} "
              f"{source_q[present].sum():.10f} {residue_field[0]:.10f} "
              f"{residue_field[1]:.10f} {residue_field[2]:.10f}")
    print("# z_center_nm charge_e field_z_contribution_V_per_nm n_sites")
    z = source_r[:, 2]
    lower = np.floor(z.min() / args.bin_width) * args.bin_width
    upper = np.ceil(z.max() / args.bin_width) * args.bin_width
    edges = np.arange(lower, upper + args.bin_width * 1.01, args.bin_width)
    indices = np.digitize(z, edges) - 1
    for bin_index in range(len(edges) - 1):
        present = indices == bin_index
        if np.any(present):
            center = (edges[bin_index] + edges[bin_index + 1]) / 2.0
            print(f"{center:.6f} {source_q[present].sum():.10f} "
                  f"{contributions[present, 2].sum():.10f} {present.sum()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
