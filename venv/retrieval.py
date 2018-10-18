#!/usr/bin/env python
from functools import reduce
from sys import stdout
from os.path import isfile, join

import requests as http

URI_ROOT = 'https://raw.githubusercontent.com/sudhof/politeness/python3/corpora'
FILES = ('wikipedia.annotated.csv', 'stack-exchange.annotated.csv')
URIS = tuple(f'{URI_ROOT}/{f}' for f in FILES)
CORPORA_DIR = join('..', 'corpora')

def pbar(msg, progress, innerwidth):
    fill = int(progress*innerwidth)
    bar = ' '
    if fill >= 1:
        bar = '#'*fill
        bar += ' '*(innerwidth-fill)
        bar = f' [{bar}] '
    return f'{msg}{bar}({progress*100:02.02f}%)...'

def corporaPaths():
    return (join(CORPORA_DIR, f) for f in FILES)

def corporaFiles():
    return filter(isfile, corporaPaths())

def haveAllCorpora():
    return len(corporaPaths()) == len(corporaFiles())

def retrieve(prn=False):
    rsps = map(http.get, URIS)
    rlen = sum(int(rsp.headers.get('content-length')) for rsp in rsps)
    dl = 0
    padding = (max(map(len, FILES)) - min(map(len, FILES))) * ' '
    for i, uri in enumerate(URIS):
        rsp = http.get(uri, stream=True)
        msg = f'Retrieving {FILES[i]}'
        with open(FILES[i], 'wb') as ostrm:
            for chunk in rsp.iter_content(chunk_size=1<<10):
                dl += len(chunk)
                progress = min(dl/rlen, 1)
                bar = pbar(msg, progress, 30)
                prn = f'\r{bar}{padding}'
                stdout.write(prn)
                ostrm.write(chunk)
        print()

if __name__ == '__main__':
    retrieve()
    print('...Done.')
