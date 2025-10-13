import numpy as np
def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b: return 1.0
    return len(a & b) / len(a | b)
