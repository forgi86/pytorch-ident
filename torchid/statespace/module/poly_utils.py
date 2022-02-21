def digits_repr(num, base, min_digits):
    res = []
    while num:
        res.append(num % base)
        num //= base

    if len(res) < min_digits:
        res = res + [0] * (min_digits - len(res))
    res.reverse()
    return res


def valid_coeffs(n_feat, p):
    n_comb = (p+1)**n_feat  # combination of possible monomials
    pows = []
    for comb in range(1, n_comb+1):
        pow = digits_repr(comb, base=p+1, min_digits=n_feat)
        if 1 < sum(pow) < p+1:
            pows.append(pow)
    return pows