# Mapping of Unicode Braille to normal letters
UNICODE_BRAILLE_DICT = {
    '⠁': 'a', '⠃': 'b', '⠉': 'c', '⠙': 'd', '⠑': 'e',
    '⠋': 'f', '⠛': 'g', '⠓': 'h', '⠊': 'i', '⠚': 'j',
    '⠅': 'k', '⠇': 'l', '⠍': 'm', '⠝': 'n', '⠕': 'o',
    '⠏': 'p', '⠟': 'q', '⠗': 'r', '⠎': 's', '⠞': 't',
    '⠥': 'u', '⠧': 'v', '⠺': 'w', '⠭': 'x', '⠽': 'y',
    '⠵': 'z', '⠶': '.', '⠲': ',', '⠐': "'", '⠠': 'capital', 
    ' ': ' '  # keep spaces
}

def unicode_braille_to_text(braille_str: str) -> str:
    """
    Converts a string of Unicode Braille characters to normal text.
    Unknown characters are replaced with '?'.
    """
    return ''.join(UNICODE_BRAILLE_DICT.get(c, '?') for c in braille_str)

# Test
if __name__ == "__main__":
    braille_input = "⠑ ⠭ ⠏ ⠇ ⠁ ⠊ ⠝"
    text_output = unicode_braille_to_text(braille_input)
    print("INPUT:", braille_input)
    print("OUTPUT:", text_output)
