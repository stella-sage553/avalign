import avalign


def test_version_is_exposed():
    assert isinstance(avalign.__version__, str)
    assert avalign.__version__.count(".") >= 2
