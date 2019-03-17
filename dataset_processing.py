
import argparse

from utils.io import *
from utils.tagging_process import *

if __name__ == "__main__":

    ## parse input
    parser = argparse.ArgumentParser()
    parser.add_argument('tagged_result')
    options = parser.parse_args()

    ## load tagging results
    data_tagged = readJSONLine(options.tagged_result)
    print('num line', len(data_tagged))

    ## replace entity with TARGET
    data_TARGET = []
    for each_line in data_tagged:
        curr_info = each_line
        if each_line['text'].strip() != 'Not Available':
            curr_tweet, curr_tags = taggingSeperate(each_line['tags'].strip())
            curr_TAR = replaceEntityTarget(each_line['curr_ner'], [i.lower() for i in curr_tweet], curr_tags)
            curr_info['text_TARGET'] = ' '.join([i for i in curr_TAR[0]])
            data_TARGET.append(curr_info)

    ## write output file
    writeJSONFile(data_TARGET, './dataset/dataset_processed.json', verbose=True)