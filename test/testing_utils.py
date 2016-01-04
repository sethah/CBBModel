

def compare_array(actual, expected, abs_tol=1e-4):
    for a, e in zip(actual, expected):
        assert abs(a - e) < abs_tol,\
            "expected %s did not equal actual %s" % (e, a)