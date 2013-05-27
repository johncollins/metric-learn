def set_default_params(s = {}, method = 'itml'):
    """
        s = SetDefaultParams(s);
        Sets default parameters
        s: user-specified parameters that are used instead of defaults
    """
    if method == 'itml':
        default_s = {
            'gamma' : 1.0,
            'beta' : 1.0,
            'const_factor' : 40.0,
            'type4_rank' : 5.0,
            'thresh' : 10e-3,
            'k' : 4,
            'max_iters' : 100000
        }
        for key in default_s:
            if key not in s:
                s[key] = default_s[key]
    return s

if __name__ == "__main__":
    s = {}
    print set_default_params(s)
    print set_default_params()
