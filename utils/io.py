

import csv
import json


def readTXTFile(path, verbose=False):
    """
    TODO: It is not the best way of reading txt files

    :param path:
    :param verbose:
    :return:
    """
    with open(path, 'r') as f:
        data = f.readlines()
    if verbose:
        print("[I] file read complete with length", len(data))
    return data


def readJSONFile(path, verbose=False):

    with open(path, "r") as f:
        data = json.load(f)
    if verbose:
        print("[I] file read complete")
    return data


def writeJSONFile(data, path, verbose=False):
    with open(path, "w") as f:
        json.dump(data, f)
    if verbose:
        print("[I] file written complete: " + path)


def readTSVFile(path, verbose=False):

    with open(path, 'r') as f:
        data = [line.strip().split('\t') for line in f]
    if verbose:
        print("[I] file read complete with length", len(data))
    return data

def readJSONLine(path, verbose=False):

    input = readTXTFile(path)

    data = []
    for each_line in input:
        each_line = each_line.strip()
        each_line = json.loads(each_line)
        data.append(each_line)
    if verbose:
        print("[I] file read complete")
    return data

def writeJSONLine(path, data, verbose=False):

    with open(path, 'w') as f:
        for i in data:
            json.dump(i, f)
            f.write('\n')

    if verbose:
        print("[I] file written complete: " + path)