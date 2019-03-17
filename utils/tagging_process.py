
def taggingSeperate(line):

    """
    split tagging results into [token], [tags] format
    :param data: results from tagging tools
    :return:

    EXAMPLE

    Input: james/B-person ball/I-person |/O citigroup/O incorporated/O |/O email/O vice/O president/O of/O web/O application/O vulnerability/O analysis/O |/O @citigrou/O .../O https://t.co/gs20umuhr3/O
    Output: ['james', 'ball', '|', 'citigroup', 'incorporated', '|', 'email', 'vice', 'president', 'of', 'web', 'application', 'vulnerability', 'analysis', '|', '@citigrou', '...', 'https:t.cogs20umuhr3'], ['B-person', 'I-person', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

    """

    tweet = []
    tags = []
    tweetData = line.strip().split(' ')
    for token in tweetData:
        splitData = token.split('/')
        datatweet = ''
        for i in range(0, len(splitData) - 1):
            datatweet += splitData[i] + ''
        tweet.append(datatweet)
        tags.append(splitData[-1])
    return tweet, tags


def getEntitySegClass(tweet, annot, lower=False, getIndices=True):

    """
    get segments containing ENTITYs
    :param tweet: a specific tweet (NEED TO BE SPLITTED)
    :param annot: corresponding tags (NEED TO BE SPLITTED)
    :param lower:
    :param getIndices:
    :return: [segs]

    ATT: input should be a single tweet

    EXAMPLE

    Input: ['james', 'ball', '|', 'citigroup', 'incorporated', '|', 'email', 'vice', 'president', 'of', 'web', 'application', 'vulnerability', 'analysis', '|', '@citigrou', '...', 'https:t.cogs20umuhr3'], ['B-person', 'I-person', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    Output: [('james ball', ((0, 2), 'B-person'))]

    """

    start = None
    result = []
    for i in range(len(tweet)):
        if "B-" in annot[i]:
            if start != None:
                if getIndices:
                    if start != len(tweet):
                        result.append((' '.join(tweet[start:i]), (start, i, annot[start])))
                    else:
                        result.append((' '.join(tweet[start:i]), (start, i, annot[start])))
                else:
                    result.append(' '.join(tweet[start:i]))
            start = i
        elif annot[i] == 'O' and start != None:
            if getIndices:
                result.append((' '.join(tweet[start:i]), ((start, i), annot[start])))
            else:
                result.append(' '.join(tweet[start:i]))
            start = None
    if start != None:
        if getIndices:
            result.append((' '.join(tweet[start:i + 1]), (start, i + 1, annot[start])))
        else:
            result.append(' '.join(tweet[start:i + 1]))
    if lower:
        if getIndices:
            result = [(x[0].lower(), x[1]) for x in result]
        else:
            result = [(x.lower()) for x in result]
    return result


def replaceEntityTarget(ent_tuple, tweet, tag):

    """
    replace ENTITY with TARGET
    :param ent: target words needed to be replaced
    :param tweet: tweet tokens (NEED TO BE SPLITTED)
    :param tag: corresponding tags (NEED TO BE SPLITTED)
    :return: tweet (target words marked as TARGET), tag (target words marked as MOD)

    ATT: input should be a single tweet, and in ent_tuple, the location for entity should be identified

    EXAMPLE

    Input: ent_tuple = ('james ball', (0, 2))
    Output: (['<TARGET>', '|', 'citigroup', 'incorporated', '|', 'email', 'vice', 'president', 'of', 'web', 'application', 'vulnerability', 'analysis', '|', '@citigrou', '...', 'https:t.cogs20umuhr3'], ['MOD', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])

    """

    def modTweetTargetEnt(tweet, ent, indices):
        start = indices[0]
        end = indices[1]
        assert ent == " ".join(tweet[start:end])
        del tweet[start:end]
        tweet.insert(start, "<TARGET>")
        return tweet

    def modTweetTarTags(tags, indices):
        start = indices[0]
        end = indices[1]
        del tags[start:end]
        tags.insert(start, "MOD")
        return tags

    # replace ENTITY with TARGET
    ent = ent_tuple[0]
    loc = ent_tuple[1]

    tweet = modTweetTargetEnt(tweet, ent, loc)
    tag = modTweetTarTags(tag, loc)
    return tweet, tag