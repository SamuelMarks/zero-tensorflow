from zero_tensorflow import strings


def test_strings_join():
    res = strings.join(["a", "b"], separator=",")
    assert str(res) == "a,b"


def test_strings_split():
    res = strings.split("a b c")
    assert list(res) == ["a", "b", "c"]


def test_strings_length():
    res = strings.length("abc")
    assert res == 3


def test_strings_regex_replace():
    res = strings.regex_replace("a b c", " ", "_")
    assert str(res) == "a_b_c"


def test_strings_regex_full_match():
    assert strings.regex_full_match("a", "a")
    assert not strings.regex_full_match("a", "b")


def test_strings_to_number():
    res = strings.to_number("1.5")
    assert res == 1.5


def test_strings_bytes_split():
    res = strings.bytes_split("abc")
    assert list(res) == ["a", "b", "c"]


def test_strings_format():
    res = strings.format("{} b {}", ["a", "c"])
    assert str(res) == "a b c"


def test_strings_lower_upper():
    assert str(strings.lower("A")) == "a"
    assert str(strings.upper("a")) == "A"


def test_strings_strip():
    assert str(strings.strip(" a ")) == "a"


def test_strings_substr():
    assert str(strings.substr("abc", 1, 1)) == "b"


def test_strings_to_hash_bucket():
    res = strings.to_hash_bucket("a", 10)
    assert 0 <= res < 10


def test_strings_stubs_exist():
    strings.ngrams([], 1)
    strings.reduce_join([])
    strings.to_hash_bucket_fast("a", 10)
    strings.to_hash_bucket_strong("a", 10, [1, 2])
    strings.unicode_decode("a", "utf-8")
    strings.unicode_decode_with_offsets("a", "utf-8")
    strings.unicode_encode("a", "utf-8")
    strings.unicode_script("a")
    strings.unicode_split("a", "utf-8")
    strings.unicode_split_with_offsets("a", "utf-8")
    strings.unicode_transcode("a", "utf-8", "utf-8")
    strings.unsorted_segment_join([], [], 1)


class MockTensor:
    def __init__(self, val):
        self.val = val

    def numpy(self):
        return self.val


def test_strings_coverage_extras():
    # Hit line 37: hasattr(val, "numpy")
    res = strings.length(MockTensor("abc"))
    assert res == 3

    # Hit line 48: if not inputs_np
    res = strings.join([])
    assert str(res) == ""

    # Hit split 0-d
    res = strings.split(MockTensor("a b"))
    assert list(res) == ["a", "b"]

    res = strings.bytes_split(MockTensor("ab"))
    assert list(res) == ["a", "b"]


def test_strings_split_array():
    res = strings.split(["a b", "c d"])
    assert len(res) == 2
    assert list(res[0]) == ["a", "b"]


def test_strings_bytes_split_array():
    res = strings.bytes_split(["ab", "cd"])
    assert len(res) == 2
    assert list(res[0]) == ["a", "b"]
