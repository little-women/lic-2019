# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-17 17:27:25
# @Last Modified by:   liwei
# @Last Modified time: 2019-04-17 20:50:07

"""
Generate files: 'train2id.txt', 'valid2id.txt', 'test2id.txt'
                'entity2id.txt', 'relation2id.txt'
train2id.txt:
            first line: the number of triples for training
            following lines: format of (e1, e2, rel)
entity2id.txt:
            all entities and corresponding ids.
"""

import sys
import os
import json
import collections
from collections import Counter


def convert_session_to_triples(filepath):

    with open(filepath, 'r', encoding='utf-8') as f:
        triples = []
        for i, line in enumerate(f):
            session = json.loads(line.strip(), encoding="utf-8",
                                 object_pairs_hook=collections.OrderedDict)
            # the triple between goals
            triples.append(session["goal"][1])
            triples.extend(session["knowledge"])

    return triples


def buildTriples(filepath, output):
    fout = open(os.path.join(output + 'all_triples.txt'),
                'w', encoding='utf-8')
    train_triples = convert_session_to_triples(
        os.path.join(filepath, 'train.txt'))
    valid_triples = convert_session_to_triples(
        os.path.join(filepath, 'dev.txt'))
    test_triples = convert_session_to_triples(
        os.path.join(filepath, 'test.txt'))

    triples = []
    triples.extend(train_triples + valid_triples + test_triples)
    for triple in triples:
        fout.write(" ".join(triple) + '\n')
    fout.close()

    return triples, train_triples, valid_triples, test_triples


def buildEntity(triples, train_triples, valid_triples, test_triples, output):
    rfout = open(os.path.join(output + 'relation2id.txt'),
                 'w', encoding='utf-8')
    efout = open(os.path.join(output + 'entity2id.txt'), 'w', encoding='utf-8')
    train_fout = open(os.path.join(output + 'train2id.txt'),
                      'w', encoding='utf-8')
    test_fout = open(os.path.join(output + 'test2id.txt'),
                     'w', encoding='utf-8')
    valid_fout = open(os.path.join(output + 'valid2id.txt'),
                      'w', encoding='utf-8')

    relations = Counter()
    entities = Counter()
    relation2id = {}
    id2relation = {}
    entity2id = {}
    id2entity = {}

    for entity1, relation, entity2 in triples:
        relations[relation] += 1
        entities[entity1] += 1
        entities[entity2] += 1

    rfout.write(str(len(relations)) + '\n')
    efout.write(str(len(entities)) + '\n')
    train_fout.write(str(len(train_triples)) + '\n')
    test_fout.write(str(len(test_triples)) + '\n')
    valid_fout.write(str(len(valid_triples)) + '\n')

    for i, relation in enumerate(relations.keys()):
        relation2id[relation] = i
        id2relation[i] = relation
        rfout.write(relation + '\t' + str(i) + '\n')
    for i, entity in enumerate(entities.keys()):
        entity2id[entity] = i
        id2entity[i] = entity
        efout.write(entity + '\t' + str(i) + '\n')

    for entity1, relation, entity2 in train_triples:
        train_fout.write(str(entity2id[entity1]) + '\t' + str(
            entity2id[entity2]) + '\t' + str(relation2id[relation]) + '\n')

    for entity1, relation, entity2 in test_triples:
        test_fout.write(str(entity2id[entity1]) + '\t' + str(
            entity2id[entity2]) + '\t' + str(relation2id[relation]) + '\n')

    for entity1, relation, entity2 in valid_triples:
        valid_fout.write(str(entity2id[entity1]) + '\t' + str(
            entity2id[entity2]) + '\t' + str(relation2id[relation]) + '\n')

    rfout.close()
    efout.close()
    train_fout.close()
    test_fout.close()
    valid_fout.close()


def main():
    triples, train_triples, valid_triples, test_triples = buildTriples(sys.argv[
                                                                       1], sys.argv[2])
    buildEntity(triples, train_triples, valid_triples,
                test_triples, sys.argv[2])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
