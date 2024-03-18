def test_load(tmpdir):
    with tmpdir.join("foo.txt").open("w") as f:
        print("Hello!", file=f)
    assert tmpdir.listdir()[0].is is None
