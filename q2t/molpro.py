#!/usr/bin/env python
# -*- coding:utf-8 -*-


class MolproError(Exception):
    pass


class Molpro(object):
    def __init__(self, logfile=None):
        self.logfile = logfile

        if self.logfile is None:
            self.log = None
        else:
            with open(self.logfile) as f:
                self.log = f.read().splitlines()

    def get_multiplicity(self):
        for line in self.log:
            line = line.lower()
            if 'spin symmetry:' in line:
                state = line.strip().split()[-1]
                states = ('singlet', 'doublet', 'triplet', 'quartet', 'quintet')
                if state in states:
                    return states.index(state) + 1
        else:
            return 1

    def get_energy(self):
        for line in reversed(self.log):
            if 'energy=' in line:
                return float(line.split()[-1])
        else:
            raise MolproError('Energy not found in {}'.format(self.logfile))
