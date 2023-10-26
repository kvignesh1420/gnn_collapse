import math
import mpmath as mp

def compute_prob():
    temp_q = 0
    temp_p = 0
    for t in range(n+1):
        temp_q += mp.power(math.comb(n, t) * mp.power(q, t) * mp.power(1-q, n - t), n)
    res = mp.power(temp_q, math.comb(C, 2) )
    return res

if __name__ == "__main__":
    N = 1000
    C = 2
    n = N//C
    a = 3.75
    b = 0.25
    p = a * math.log(N)/N
    q = b * math.log(N)/N
    print(n, p, q)
    print("ERec: sqrt(a) - sqrt(b) = ", math.sqrt(a) - math.sqrt(b))
    print(compute_prob())
