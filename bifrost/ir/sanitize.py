def sanitize(in_text: str) -> str:
    _invalid = ['.', '-', ' ']
    sanitized_text = "".join(['_' if chr in _invalid else chr for chr in in_text])
    return sanitized_text