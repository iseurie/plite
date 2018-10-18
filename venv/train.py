#!/usr/bin/env python

from os.path import join
import random as rnd

import fasttext
import retrieval
import gentags as tag

MODEL_DIR = join('..', 'model')
DATA_DIR = join('..', 'data')

args = {
    'wordNgrams': 1,
    'lr': 0.1,
    'threads': 3,
    'lrUpdateRate': 100,
    'dim': 50,
    'bucket': 40000,
    'epoch': 10}

def train(oname=None):
    oname = oname or 'politeness.full'
    corpus = join('data', oname)
    opath = join(MODEL_DIR, oname)
    if not retrieval.haveAllCorpora():
        retrieval.retrieve(True)
    print(f'[{opath}] Serializing annotations...')
    tag.Annotations(tag.spool()).writeTo(corpus)
    return fasttext.supervised(input=corpus, output=opath, **args)

def validate(alpha):
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
    validate(0.2)
    train()
