import logging
import itertools
import numpy as np
from collections.abc import Iterable

chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r','s', 't', 'u',
         'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '!', '?', ',', '.','\'','"', '-',':',';' ,
         '(', ')', '%','/', '—', '–', '”', '“', '+', '=', '§', '[', ']', '’', '&', '*', '#']


class TextMapper:
    def __init__(self, chars, joker='�', pad='Ξ', char_mapping=[]):
        self.chars = chars
        self.chars = [joker, pad] + self.chars
        self._char_mapping = char_mapping
        self._from_char = list(self.chars)
        self.joker = joker
        self.pad = pad
        self._joker_position = self._from_char.index(joker)
        self._to_char = [chr(i) for i in range(len(self.chars))]
        self._translation_table = None
        self._update_translation_table()

    def _update_translation_table(self):
        self._translation_table = str.maketrans(''.join(self._from_char), ''.join(self._to_char))

    def map_text(self, text):
        return (self._char_mapping[char] if char in self._char_mapping else char for char in text)

    def texts_to_labels(self, texts):
        texts = [tuple(self.map_text(text)) for text in texts]
        missing_chars = set(''.join(itertools.chain(*texts))) - set(self._from_char)
        if missing_chars:
            for char in missing_chars:
                self._from_char.append(char)
                self._to_char.append(chr(self._joker_position))

            self._update_translation_table()

        labels = [np.asarray([ord(x) for x in ''.join(text).translate(self._translation_table)], dtype=np.int32)
                  for text in texts]
        return labels

    def string_to_labels(self, s: str):
        labels = self.texts_to_labels([s])
        return labels[0]

    def labels_to_text(self, labels):
        if type(labels[0]) == int:
            text = ''.join([self._from_char[l] for l in labels])
        elif isinstance(labels[0], Iterable):
            text = [''.join([self._from_char[l] for l in line_labels]) for line_labels in labels]
        else:
            raise ValueError(f'Unrecognized labels type ({type(labels)}). It should be 1D or 2D list of ints.')

        return text
