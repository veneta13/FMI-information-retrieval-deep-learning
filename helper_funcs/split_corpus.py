import random


def split_class_corpus(full_class_corpus, test_fraction=0.1):
    test_class_corpus = []
    train_class_corpus = []
    random.seed(42)
    for c in range(len(full_class_corpus)):
        class_list = full_class_corpus[c]
        random.shuffle(class_list)
        test_count = int(len(class_list) * test_fraction)
        test_class_corpus.append(class_list[:test_count])
        train_class_corpus.append(class_list[test_count:])
    return test_class_corpus, train_class_corpus
