#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import json
import os
import shutil

from q2t.thermo import get_thermo
from q2t.qchem import QChem
from q2t.mol import str_to_rmg_mol, geo_to_rmg_mol, geo_to_xyz_str


def main():
    args = parse_args()

    if args.bacs is not None:
        with open(args.bacs) as f:
            bacs = json.load(f)
    else:
        bacs = None

    with open(args.identifier_file) as f:
        identifiers = [line.strip() for line in f if line.strip()]
    if args.exceptions_file is not None:
        with open(args.exceptions_file) as f:
            exceptions = {line.strip() for line in f if line.strip()}
    else:
        exceptions = set()

    if args.energy_dir is None:
        optfreq_logs = {int(os.path.basename(log).split('.')[0]): log
                        for log in glob.iglob(os.path.join(args.optfreq_dir, '[0-9]*.out'))}
        energy_logs = {i: None for i in optfreq_logs}
    else:
        energy_logs = {int(os.path.basename(log).split('.')[0]): log
                       for log in glob.iglob(os.path.join(args.energy_dir, '[0-9]*.out'))}
        optfreq_logs = {i: os.path.join(args.optfreq_dir, '{}.out'.format(i)) for i in energy_logs}

    out_dir = args.out_dir
    scr_dir = os.path.join(out_dir, 'SYMM_SCRATCH')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(scr_dir):
        os.mkdir(scr_dir)

    thermo, geos = {}, {}
    for i, energy_log in energy_logs.iteritems():
        optfreq_log = optfreq_logs[i]
        identifier = identifiers[i]
        try:
            geo = QChem(logfile=optfreq_log).get_geometry()
        except IOError:
            print('Missing optfreq file {}!'.format(optfreq_log))
            continue
        mol_check = str_to_rmg_mol(identifier, single_bonds=True)
        mol = geo_to_rmg_mol(geo)
        if identifier in exceptions or mol_check.isIsomorphic(mol):
            thermo[identifier] = get_thermo(optfreq_log, args.freq_level, args.model_chemistry, energy_log=energy_log,
                                            mol=mol, bacs=bacs, soc=args.soc,
                                            infer_symmetry=args.symmetry, infer_chirality=args.chirality,
                                            unique_id=str(i), scr_dir=scr_dir)
            geos[identifier] = geo
        else:
            print('Ignored {}: {} does not match parsed geometry!'.format(optfreq_log, identifier))

    shutil.rmtree(scr_dir)

    hpath = os.path.join(out_dir, 'hf298.csv')
    spath = os.path.join(out_dir, 's298.csv')
    cpath = os.path.join(out_dir, 'cp.csv')
    gpath = os.path.join(out_dir, 'geos.xyz')
    with open(hpath, 'w') as hf, open(spath, 'w') as sf, open(cpath, 'w') as cf, open(gpath, 'w') as gf:
        for identifier in thermo:
            hf298, s298, cp = thermo[identifier]
            geo = geos[identifier]
            hf.write('{}   {}\n'.format(identifier, hf298))
            sf.write('{}   {}\n'.format(identifier, s298))
            cf.write('{0}   {1[0]} {1[1]} {1[2]} {1[3]} {1[4]} {1[5]} {1[6]}\n'.format(identifier, cp))
            gf.write(geo_to_xyz_str(geo))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('identifier_file', help='File containing molecule identifiers in order')
    parser.add_argument('optfreq_dir', help='Directory containing Q-Chem optfreq jobs')
    parser.add_argument('out_dir', help='Output directory')
    parser.add_argument('--energy_dir', help='Directory containing Molpro energy jobs (optional)')
    parser.add_argument('--model_chemistry', default='ccsd(t)-f12a/cc-pvdz-f12', help='Level of theory for energy')
    parser.add_argument('--freq_level', default='wb97x-d3/def2-tzvp', help='Level of theory for frequencies')
    parser.add_argument('--soc', action='store_true', help='Use spin-orbit corrections')
    parser.add_argument('--exceptions_file', help='File containing molecule identifiers that override '
                                                  'match checking of true and parsed geometry')
    parser.add_argument('--bacs', help='.json file containing BACs in kcal/mol')
    parser.add_argument('--symmetry', action='store_true', help='Infer symmetry')
    parser.add_argument('--chirality', action='store_true', help='Infer chirality')
    return parser.parse_args()


if __name__ == '__main__':
    main()
