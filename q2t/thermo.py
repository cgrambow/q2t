#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

import rmgpy.constants as constants
from rmgpy.statmech import Conformer, IdealGasTranslation, LinearRotor, NonlinearRotor, HarmonicOscillator
from rmgpy.qm.qmdata import QMData
from rmgpy.qm.symmetry import PointGroupCalculator

from .qchem import QChem
from .molpro import Molpro
from .mol import atomic_symbol_dict, geo_to_rmg_mol, get_bac_correction

# Experimental heats of formation of atoms (kcal/mol)
h0expt = {'H': 51.63,
          'C': 169.98,
          'N': 112.53,
          'O': 58.99}
h298corr = {'H': 1.01,
            'C': 0.25,
            'N': 1.04,
            'O': 1.04}

# Atomic reference energies at 0K in Hartree
atom_energies = {
    'ccsd(t)-f12a/cc-pvdz-f12': {
        'H': -0.499811124128,
        'N': -54.525946786123,
        'O': -74.994643838203,
        'C': -37.787831744881,
    },
    'ccsd(t)-f12b/cc-pvdz-f12': {
        'H': -0.499811124128,
        'N': -54.522814689877,
        'O': -74.989919455883,
        'C': -37.785040449664,
    },
    'ccsd(t)-f12a/cc-pvtz-f12': {
        'H': -0.499946213253,
        'N': -54.529590447091,
        'O': -75.003545717458,
        'C': -37.789552049511,
    },
    'ccsd(t)-f12b/cc-pvtz-f12': {
        'H': -0.499946213253,
        'N': -54.527721253368,
        'O': -75.000516530163,
        'C': -37.787925879006,
    },
    'ccsd(t)-f12a/cc-pvqz-f12': {
        'H': -0.499994558326,
        'N': -54.530194782830,
        'O': -75.005192195863,
        'C': -37.789729174726,
    },
    'ccsd(t)-f12b/cc-pvqz-f12': {
        'H': -0.499994558326,
        'N': -54.529107245074,
        'O': -75.003414816890,
        'C': -37.788775207449,
    },
    'ccsd(t)-f12a/aug-cc-pv5z': {
        'H': -0.499994816870,
        'N': -54.529731561126,
        'O': -75.004562049197,
        'C': -37.789360554007,
    },
    'ccsd(t)-f12b/aug-cc-pv5z': {
        'H': -0.499994816870,
        'N': -54.528933245046,
        'O': -75.003291308092,
        'C': -37.788641170961,
    },
    'b3lyp/6-31g(2df,p)': {
        'H': -0.500273,
        'N': -54.583861,
        'O': -75.064579,
        'C': -37.846772,
    }
}

freq_scale_factors = {
    'b3lyp/6-31g(2df,p)': 0.965,
    'wb97x-d3/def2-tzvp': 0.975,
}


def get_thermo(optfreq_log, optfreq_level, energy_level, energy_log=None,
               mol=None, bacs=None,
               infer_symmetry=False, infer_chirality=False, unique_id='0', scr_dir='SCRATCH'):

    q = QChem(logfile=optfreq_log)
    symbols, coords = q.get_geometry()
    inertia = q.get_moments_of_inertia()
    freqs = q.get_frequencies()
    zpe = q.get_zpe()

    if energy_log is None:
        e0 = q.get_energy()
        multiplicity = q.get_multiplicity()
    else:
        m = Molpro(logfile=energy_log)
        e0 = m.get_energy()
        multiplicity = m.get_multiplicity()

    # Infer connections only if not given explicitly
    if mol is None:
        mol = geo_to_rmg_mol((symbols, coords))  # Does not contain bond orders

    # Try to infer point group to calculate symmetry number and chirality
    symmetry = optical_isomers = 1
    point_group = None
    if infer_symmetry or infer_chirality:
        qmdata = QMData(
            groundStateDegeneracy=multiplicity,  # Only needed to check if valid QMData
            numberOfAtoms=len(symbols),
            atomicNumbers=[atomic_symbol_dict[sym] for sym in symbols],
            atomCoords=(coords, 'angstrom'),
            energy=(e0 * 627.5095, 'kcal/mol')  # Only needed to avoid error
        )
        settings = type("", (), dict(symmetryPath='symmetry', scratchDirectory=scr_dir))()  # Creates anonymous class
        pgc = PointGroupCalculator(settings, unique_id, qmdata)
        point_group = pgc.calculate()
    if point_group is not None:
        if infer_symmetry:
            symmetry = point_group.symmetryNumber
        if infer_chirality and point_group.chiral:
            optical_isomers = 2

    # Translational mode
    mass = mol.getMolecularWeight()
    translation = IdealGasTranslation(mass=(mass, 'kg/mol'))

    # Rotational mode
    if isinstance(inertia, list):  # Nonlinear
        rotation = NonlinearRotor(inertia=(inertia, 'amu*angstrom^2'), symmetry=symmetry)
    else:
        rotation = LinearRotor(inertia=(inertia, 'amu*angstrom^2'), symmetry=symmetry)

    # Vibrational mode
    freq_scale_factor = freq_scale_factors.get(optfreq_level, 1.0)
    freqs = [f * freq_scale_factor for f in freqs]
    vibration = HarmonicOscillator(frequencies=(freqs, 'cm^-1'))

    # Bring energy to gas phase reference state
    e0 *= constants.E_h * constants.Na
    zpe *= constants.E_h * constants.Na * freq_scale_factor
    for sym in symbols:
        e0 -= atom_energies[energy_level][sym] * constants.E_h * constants.Na
        e0 += (h0expt[sym] - h298corr[sym]) * 4184.0

    if bacs is not None:
        e0 -= get_bac_correction(mol, **bacs) * 4184.0

    # Group modes into Conformer object
    modes = [translation, rotation, vibration]
    conformer = Conformer(modes=modes, spinMultiplicity=multiplicity, opticalIsomers=optical_isomers)

    # Calculate heat of formation, entropy of formation, and heat capacities
    conformer.E0 = (e0 + zpe, 'J/mol')
    hf298 = conformer.getEnthalpy(298.0) + conformer.E0.value_si
    s298 = conformer.getEntropy(298.0)

    Tlist = [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0]
    cp = np.zeros(len(Tlist))
    for i, T in enumerate(Tlist):
        cp[i] = conformer.getHeatCapacity(T)

    # Return in kcal/mol and cal/mol/K
    return hf298/4184.0, s298/4.184, cp/4.184
