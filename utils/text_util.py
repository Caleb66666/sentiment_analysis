from hanziconv import HanziConv


def t2s(text):
    return HanziConv.toSimplified(text)


def full2half(text):
    def char2half(char):
        uchar = ord(char)
        if uchar == 12288:
            return chr(32)
        if uchar in range(65281, 65375):
            return chr(uchar - 65248)
        return char

    return "".join([char2half(char) for char in text])


def rm_blank(text):
    return "".join(text.split())


def rm_quote(text):
    while text.startswith('"'):
        text = text[1:]
    while text.endswith('"'):
        text = text[:-1]
    return text


def pretreatment(text):
    text = rm_blank(text)
    text = rm_quote(text)
    text = t2s(text)
    text = full2half(text)
    return text
