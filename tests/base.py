# -*- coding: utf-8 -*-

import numbers

import numpy as np
import pytest


def assert_nested_equal(actual, expected, rel=1e-12, abs=1e-14):
    """
    A helper function for confirming the results in two json files are the same
    within floating point tolerance.
    """

    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()

        for key in expected:
            assert_nested_equal(actual[key], expected[key], rel=rel, abs=abs)

    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)

        for a, e in zip(actual, expected):
            assert_nested_equal(a, e, rel=rel, abs=abs)

    elif isinstance(expected, numbers.Real):
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=rel, abs=abs)

    else:
        assert actual == expected
