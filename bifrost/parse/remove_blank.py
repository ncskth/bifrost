from copy import copy

def remove_blank(text: str) -> str:
    t = copy(text)
    return t.replace("\n", "").replace(" ", "")