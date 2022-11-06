import math


def select_features_by_frequency(V, count):
    if count > len(V): count = len(V)
    VLen = [(t, sum(V[t])) for t in V]
    VLenSorted = sorted(VLen, key=lambda x: x[1])
    return set([x[0] for x in VLenSorted[-count:]])


def select_features_by_MI(Nc, V, count):
    N = sum(Nc)
    C_lists = []
    for c in range(len(Nc)):
        VMI = []
        for t in V:
            N11 = V[t][c]
            N10 = sum(V[t]) - N11
            N01 = Nc[c] - N11
            N00 = N - Nc[c] - N10
            N1_ = N11 + N10
            N0_ = N01 + N00
            N_0 = N00 + N10
            N_1 = N01 + N11
            MI = ((N11 + 1.0) / (N + 4.0)) * math.log(((N + 4.0) * (N11 + 1.0)) / ((N1_ + 2.0) * (N_1 + 2.0)), 2) + (
                    (N01 + 1.0) / (N + 4.0)) * math.log(((N + 4.0) * (N01 + 1.0)) / ((N0_ + 2.0) * (N_1 + 2.0)),
                                                        2) + ((N10 + 1.0) / (N + 4.0)) * math.log(
                ((N + 4.0) * (N10 + 1.0)) / ((N1_ + 2.0) * (N_0 + 2.0)), 2) + ((N00 + 1.0) / (N + 4.0)) * math.log(
                ((N + 4.0) * (N00 + 1.0)) / ((N0_ + 2.0) * (N_0 + 2.0)), 2)
            VMI.append((t, MI))
        VMIsorted = sorted(VMI, key=lambda x: x[1], reverse=True)
        C_lists.append(VMIsorted)

    result = set()
    for i in range(len(V)):
        for c in range(len(Nc)):
            result.add(C_lists[c][i][0])
            if len(result) == count: break
        if len(result) == count: break

    return result
