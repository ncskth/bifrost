def sanitize(txt: str) -> str:
    _invalid = ['.', '-', ' ']
    stxt = "".join(['_' if chr in _invalid else chr for chr in txt])
    return stxt