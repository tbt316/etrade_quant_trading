import numpy as np
def check_w(d, min_weight=1e-4):
    w = [1.0]
    k = 1
    while True:
        next_w = -w[-1] * (d - k + 1) / k
        if abs(next_w) < min_weight:
            break
        w.append(next_w)
        k += 1
    return len(w)

for d in [0.2, 0.5, 0.8, 1.0]:
    print(f"d={d}, len(w)={check_w(d)}")
