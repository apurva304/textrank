"""Python implementation of the TextRank algoritm.

From this paper:
    https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

Based on:
    https://gist.github.com/voidfiles/1646117
    https://github.com/davidadamojr/TextRank
"""
import io
import itertools
import networkx as nx
import nltk
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    #print('in stem_tokens')
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    #print('in normalize')
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def setup_environment():
    """Download required resources."""
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    print('Completed resource downloads.')


'''def levenshtein_distance(first, second):
    """Return the Levenshtein distance between two strings.

    Based on:
        http://rosettacode.org/wiki/Levenshtein_distance#Python
    """
    if len(first) > len(second):
        first, second = second, first
    distances = range(len(first) + 1)
    for index2, char2 in enumerate(second):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(first):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1],
                                              distances[index1 + 1],
                                              new_distances[-1])))
        distances = new_distances
    return distances[-1]'''


def build_graph(nodes):
    """Return a networkx graph instance.

    :param nodes: List of hashables that represent the nodes of a graph.
    """
    gr = nx.Graph()  # initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))
    l = len(nodePairs)

    printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)
    for i, pair in enumerate(nodePairs):
        firstString = pair[0]
        secondString = pair[1]
        #print('firstString:\t' + pair[0])
        #print('secondString:\t' + pair[1])
        tfdif = cosine_sim(firstString, secondString)
        #print('The Edge is:\t',tfdif)
        gr.add_edge(firstString, secondString, weight=tfdif)
        printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=50)

    nx.draw(gr,with_labels=True)
    plt.show()
    return gr


def extract_sentences(text, summary_length=120, clean_sentences=False, language='english'):
    """Return a paragraph formatted summary of the source text.

    :param text: A string.
    """
    print("tokenizing.......")
    sent_detector = nltk.data.load('tokenizers/punkt/' + language + '.pickle')
    sentence_tokens = sent_detector.tokenize(text.strip())

    #print(sentence_tokens)

    print("building graph")
    graph = build_graph(sentence_tokens)

    print("calculating ranks")
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,
                       reverse=True)

    # return a 100 word summary
    summary = ' '.join(sentences)
    summary_words = summary.split()
    summary_words = summary_words[0:summary_length]
    dot_indices = [idx for idx, word in enumerate(summary_words) if word.find('.') != -1]
    if clean_sentences and dot_indices:
        last_dot = max(dot_indices) + 1
        summary = ' '.join(summary_words[0:last_dot])
    else:
        summary = ' '.join(summary_words)

        print("\nGenerated summary\n")
        print("-" * 50)
    return summary


print("1 : initial Setup required first time only")
print("2 : skipt initial setup")

user_input = int(input("Enter your choice\n"))
if user_input == 1:
    print("checking for updates")
    setup_environment()
    file = input("Enter the file name\n")
    with open(file) as f:
        summary = extract_sentences(f.read())
        print(summary)
elif user_input == 2:
    filename = input("Enter the file name\n")
    with open(filename) as f:
        summary = extract_sentences(f.read())
        print(summary)
