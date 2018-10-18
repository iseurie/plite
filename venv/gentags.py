#!/usr/bin/env python
import re
import csv
import io
import unicodedata
import os.path
from random import shuffle
import numpy.random as npr

from scipy.stats import norm
import retrieval

def zbin(z: float) -> int:
    '''
    Take a z-score normalization value and return a uniformly-distributed
    integer on the interval [0, 6].
    '''
    return int(norm.sf(z)*6)

def sanitize(s, labelPrefix='__label__'):
    ucat = unicodedata.category
    s = unicodedata.normalize('NFD', s)
    s = ''.join('' if ucat(c) == 'Mn'\
                else f' {c} '\
                if ucat(c)[0] == 'P'\
                    and c != '@'\
                    and (True if i == 0 else s[i-1].isalnum())\
                    and (True if i == len(s)-1 else s[i+1].isalnum())\
                else c for i, c in enumerate(s))
    s = re.sub(r'\s{2,}', ' ', s)
    s = re.sub(r'(.)\1+', r'\1\1', s)
    s = re.sub(labelPrefix, '', s)
    s.lower()
    return s

DEFAULT_LABEL_PREFIX = '__label__'

def spool(files=retrieval.corporaFiles(), labelPrefix=DEFAULT_LABEL_PREFIX):
    for ipath in files:
        with open(ipath, 'r') as istrm:
            rd = csv.reader(istrm)
            rows = iter(rd)
            # skip header
            next(rows)
            for row in rows:
                content = sanitize(row[2], labelPrefix)
                z = float(row[-1])
                cat = zbin(z)
                yield (cat, content)

def shuffleSpool(files=retrieval.corporaFiles(), labelPrefix=DEFAULT_LABEL_PREFIX, chunksz=1<<12):
    istreams = tuple(io.BufferedReader(open(f, 'r')) for f in files)
    lengths = tuple(map(os.path.getsize, files))
    breaks = [[0] for _ in range(len(files))]
    breakcs = list(map(len, breaks))
    breakc = sum(breakcs)
    # probability sampling from files; scale by n. lines
    # to yield a uniform distribution over all lines
    P = tuple(s/breakc for s in breakcs)

    for i, istrm in enumerate(istreams):
        for chunk in iter(istrm.read(chunksz), ''):
            offset = istrm.tell() - len(chunk)
            breaks[i] += tuple(i+offset for i, b in enumerate(chunk)
                               # ensure that we don't add the last newline, which is
                               # problematic if we seek to it expecting to read the
                               # subsequent line
                               if b == '\n' and i+offset != lengths[i]-1)
    # randomize our itinerary
    for indices in breaks:
        shuffle(indices)

    while breakc > 0:
        strm, indices = npr.choice(enumerate(breaks), 1, P)
        offset = npr.choice(indices, 1, replace=False)
        breakcs[strm] -= 1
        breakc -= 1
        # reinitialize the probability distribution
        P = tuple(s/breakc for s in breakcs)
        # give the caller their random line
        yield istreams[strm].seek(offset).readline()

    for istream in istreams:
        istream.close()

class Annotations(object):
    def __init__(self, observations=tuple()):
        self.clear()
        self.unspool(observations)

    def clear(self):
        self.observations = tuple([] for _ in range(6))

    def unspool(self, spool):
        self.clear()
        for cat, entry in spool:
            self.observations[cat].append(entry)
        return self

    def spool(self):
        for cat, entries in enumerate(self.observations):
            for entry in entries:
                yield (cat, entry)

    def writeTo(self, opath, labelPrefix=DEFAULT_LABEL_PREFIX):
        with open(opath, 'w') as ostrm:
            gen = (f'{labelPrefix}{cat} {content}'
                   for cat, content in self.spool())
            ostrm.writelines(gen)
