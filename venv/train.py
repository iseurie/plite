#!/usr/bin/env python

from os.path import join
import random as rnd

from itertools import tee
import fasttext
import retrieval
import gentags as tag

MODEL_DIR = join('..', 'model')

args = {
    'wordNgrams': 1,
    'pretrainedVectors': join(retrieval.CORPORA_DIR, 'slim.vec'),
    'lr': 0.1,
    'threads': 12,
    'lrUpdateRate': 100,
    'bucket': 200000,
    'epoch': 10}

def train(oname=None):
    oname = oname or 'politeness.full'
    corpus = join('data', oname)
    opath = join(MODEL_DIR, oname)
    if len(tuple(retrieval.missing())) > 0:
        retrieval.retrieve(True)
    
    print(f'[{opath}] Serializing annotations...')
    tag.Annotations(tag.spool()).writeTo(corpus)
    return fasttext.supervised(input=corpus, output=opath, **args)

def validate(alpha):
    if len(tuple(retrieval.missing())) > 0:
        retrieval.retrieve(True)
    ratings = list(tag.spool())
    rnd.shuffle(ratings)
    i = int(len(ratings)*alpha)
    vtags = tag.Annotations(ratings[:i-1])
    ttags = tag.Annotations(ratings[i:])
    training, testing = (join(MODEL_DIR, p) for p in
                         ('politeness.validation.training.txt',
                          'politeness.validation.testing.txt'))
    vtags.writeTo(training)
    ttags.writeTo(testing)
    fasttext.supervised(training, output=training[:-4])
    fasttext.supervised(training, output=testing[:-4])
    validation = fasttext.load_model(training)
    test = validation.test(testing)
    print('Test results\n'
          f'P@1: {test.precision}\n'
          f'R@1: {test.recall}\n')

if __name__ == '__main__':
    from sys import argv
    f1, f2 = tee(argv[1:])
    next(f2, None)
    # pass on cfg. parameters specified on the command-line
    args.update({k[1:]: v for k, v in zip(f1, f2)
                 if k[0] == '-'})

    validate(0.2)
    train()
