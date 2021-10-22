from copy import copy

TAB = " " * 4

def remove_blank(text: str) -> str:
    t = copy(text)
    return t.replace("\n", "").replace(" ", "")


def sanitize(in_text: str) -> str:
    _invalid = ['.', '-', ' ']
    sanitized_text = "".join(['_' if chr in _invalid else chr for chr in in_text])
    return sanitized_text