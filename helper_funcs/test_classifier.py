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


def test_classifier(test_class_corpus, gamma):
    L = [len(c) for c in test_class_corpus]
    pb = ProgressBar(50)
    pb.start(sum(L))
    i = 0
    classes_count = len(test_class_corpus)
    confusion_matrix = [[0] * classes_count for _ in range(classes_count)]
    for c in range(classes_count):
        for text in test_class_corpus[c]:
            pb.tick(i)
            i += 1
            c_MAP = gamma(text)
            confusion_matrix[c][c_MAP] += 1

    pb.stop()
    precision = []
    recall = []
    F_score = []

    for c in range(classes_count):
        extracted = sum(confusion_matrix[x][c] for x in range(classes_count))
        if confusion_matrix[c][c] == 0:
            precision.append(0.0)
            recall.append(0.0)
            F_score.append(0.0)
        else:
            precision.append(confusion_matrix[c][c] / extracted)
            recall.append(confusion_matrix[c][c] / L[c])
            F_score.append((2.0 * precision[c] * recall[c]) / (precision[c] + recall[c]))

    P = sum(L[c] * precision[c] / sum(L) for c in range(classes_count))
    R = sum(L[c] * recall[c] / sum(L) for c in range(classes_count))
    F1 = (2 * P * R) / (P + R)
    return confusion_matrix, precision, recall, F_score, P, R, F1
