#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from six.moves import xrange

import numpy as np


class QChemError(Exception):
    pass


class QChem(object):
    def __init__(self, logfile=None):
        self.logfile = logfile

        if logfile is None:
            self.log = None
        else:
            with open(logfile) as f:
                self.log = f.read().splitlines()
                for line in self.log:
                    if 'fatal error' in line:
                        raise QChemError('Q-Chem job {} had an error!'.format(logfile))

    def get_energy(self):
        for line in reversed(self.log):
            if 'SCF failed to converge' in line:
                raise QChemError('SCF failed to converge in {}'.format(self.logfile))
            elif 'total energy' in line:  # Double hybrid methods
                return float(line.split()[-2])
            elif 'energy in the final basis set' in line:  # Other DFT methods
                return float(line.split()[-1])
        else:
            raise QChemError('Energy not found in {}'.format(self.logfile))

    def get_geometry(self):
        for i in reversed(xrange(len(self.log))):
            line = self.log[i]
            if 'SCF failed to converge' in line:
                raise QChemError('SCF failed to converge in {}'.format(self.logfile))
            elif 'Standard Nuclear Orientation' in line:
                symbols, coords = [], []
                for line in self.log[(i+3):]:
                    if '----------' not in line:
                        data = line.split()
                        symbols.append(data[1])
                        coords.append([float(c) for c in data[2:]])
                    else:
                        return symbols, np.array(coords)
        else:
            raise QChemError('Geometry not found in {}'.format(self.logfile))

    def get_moments_of_inertia(self):
        for line in reversed(self.log):
            if 'SCF failed to converge' in line:
                raise QChemError('SCF failed to converge in {}'.format(self.logfile))
            elif 'Eigenvalues --' in line:
                inertia = [float(i) * 0.52917721092**2.0 for i in line.split()[-3:]]  # Convert to amu*angstrom^2
                if inertia[0] == 0.0:  # Linear rotor
                    inertia = np.sqrt(inertia[1]*inertia[2])
                return inertia

    def get_frequencies(self):
        freqs = []
        for line in reversed(self.log):
            if 'SCF failed to converge' in line:
                raise QChemError('SCF failed to converge in {}'.format(self.logfile))
            elif 'Frequency' in line:
                freqs.extend([float(f) for f in reversed(line.split()[1:])])
            elif 'VIBRATIONAL ANALYSIS' in line:
                freqs.reverse()
                return np.array(freqs)
        else:
            raise QChemError('Frequencies not found in {}'.format(self.logfile))

    def get_zpe(self):
        for line in reversed(self.log):
            if 'SCF failed to converge' in line:
                raise QChemError('SCF failed to converge in {}'.format(self.logfile))
            elif 'Zero point vibrational energy' in line:
                return float(line.split()[-2]) / 627.5095  # Convert to Hartree
        else:
            raise QChemError('ZPE not found in {}'.format(self.logfile))

    def get_multiplicity(self):
        for i, line in enumerate(self.log):
            if '$molecule' in line:
                return int(self.log[i + 1].strip().split()[-1])
        else:
            raise QChemError('Multiplicity not found in {}'.format(self.logfile))
