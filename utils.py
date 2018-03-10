import math
import time
import unicodedata

import matplotlib
import torch
from torch.autograd import Variable

from language import Language

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Sentence Max Length
max_length = 40


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("data/nmt-loss.jpeg")


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    # print("normalize: {}".format(s))
    # print("nor-unicode bf {} aft {}".format(_s,s))
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # print("Final String: {}".format(s))
    return s


# Turns a unicode string to plain ASCII (http://stackoverflow.com/a/518232/2809427)
def unicode_to_ascii(s):
    chars = [c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn']
    char_list = ''.join(chars)
    return char_list



def filter_pair(p):
    is_good_length = len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length
    return is_good_length


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang_name, _dir):
    # Read and filter sentences
    input_lang, output_lang, pairs = read_languages(lang_name, _dir)
    pairs = filter_pairs(pairs)
    
    # Index words
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])
    
    return input_lang, output_lang, pairs


def read_languages(lang, _dir):
    # Read and parse the text file
    doc = open(_dir + '/%s.txt' % lang).read()
    lines = doc.strip().split('\n')
    
    # Transform the data and initialize language instances
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    input_lang = Language('eng')
    output_lang = Language(lang)
    return input_lang, output_lang, pairs


# Returns a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(1)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    var = var
    return var


def variables_from_pair(pair, input_lang, output_lang):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return input_variable, target_variable
