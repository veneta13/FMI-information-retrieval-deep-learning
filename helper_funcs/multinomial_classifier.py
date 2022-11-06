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


def train_multinomial_NB(train_class_corpus):
    N = sum(len(classList) for classList in train_class_corpus)
    classes_count = len(train_class_corpus)
    pb = ProgressBar(50)
    pb.start(N)
    V = {}
    i = 0
    for c in range(classes_count):
        for text in train_class_corpus[c]:
            pb.tick(i)
            i += 1
            terms = [token.lower() for token in text if token.isalpha()]
            for term in terms:
                if term not in V:
                    V[term] = [0] * classes_count
                V[term][c] += 1
    pb.stop()

    Nc = [(len(classList)) for classList in train_class_corpus]
    prior = [Nc[c] / N for c in range(classes_count)]
    T = [0] * classes_count
    for t in V:
        for c in range(classes_count):
            T[c] += V[t][c]
    condProb = {}
    for t in V:
        condProb[t] = [(V[t][c] + 1) / (T[c] + len(V)) for c in range(classes_count)]
    return condProb, prior, V


def apply_multinomial_NB(prior, cond_prob, text, features=None):
    terms = [token.lower() for token in text if token.isalpha()]
    for c in range(len(prior)):
        score = math.log(prior[c])
        for t in terms:
            if t not in cond_prob: continue
            if features and t not in features: continue
            score += math.log(cond_prob[t][c])
        if c == 0 or score > max_score:
            max_score = score
            answer = c
    return answer
