import math
import sys

class ProgressBar:
    def __init__(self, bar_width=50):
        self.bar_width = bar_width
        self.period = None

    def start(self, count):
        self.period = int(count / self.bar_width)
        sys.stdout.write("[" + (" " * self.bar_width) + "]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.bar_width + 1))

    def tick(self, item):
        if item > 0 and item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()

    def stop(self):
        sys.stdout.write("]\n")


def train_bernoulli_NB(train_class_corpus):
    N = sum(len(class_list) for class_list in train_class_corpus)
    classes_count = len(train_class_corpus)
    pb = ProgressBar(50)
    pb.start(N)
    V = {}
    i = 0

    for c in range(classes_count):
        for text in train_class_corpus[c]:
            pb.tick(i)
            i += 1
            terms = set([token.lower() for token in text if token.isalpha()])
            for term in terms:
                if term not in V:
                    V[term] = [0] * classes_count
                V[term][c] += 1
    pb.stop()

    Nc = [len(classList) for classList in train_class_corpus]
    prior = [Nc[c] / N for c in range(classes_count)]
    cond_prob = {}
    for t in V:
        cond_prob[t] = [(V[t][c] + 1) / (Nc[c] + 2) for c in range(classes_count)]

    return cond_prob, prior, V


def apply_bernoulli_NB_SLOW(prior, cond_prob, text, features=None):
    terms = set([token.lower() for token in text if token.isalpha()])

    for c in range(len(prior)):
        score = math.log(prior[c])
        for t in cond_prob:
            if features and t not in features: continue
            if t in terms:
                score += math.log(cond_prob[t][c])
            else:
                score += math.log(1.0 - cond_prob[t][c])

        if c == 0 or score > max_score:
            max_score = score
            answer = c

    return answer


def calc_initial_cond_prob(cond_prob, features=None):
    classesCount = len(cond_prob[next(iter(cond_prob))])
    initialCondProb = [0.0] * classesCount

    for t in features if features else cond_prob:
        for c in range(classesCount):
            initialCondProb[c] += math.log(1.0 - cond_prob[t][c])

    return initialCondProb


def apply_bernoulli_NB(prior, cond_prob, initial_cond_prob, text, features=None):
    terms = set([token.lower() for token in text if token.isalpha()])

    for c in range(len(prior)):
        score = math.log(prior[c]) + initial_cond_prob[c]
        for t in terms:
            if t not in cond_prob: continue
            if features and t not in features: continue
            score += math.log(cond_prob[t][c] / (1.0 - cond_prob[t][c]))
        if c == 0 or score > max_score:
            max_score = score
            answer = c

    return answer
