import os
import numpy as np
import torch
import random

from typing import List

from text_mapper import TextMapper
from text_mapper import chars


class SpellDataset:
    def __init__(self, txt_path, min_err=0, max_err=5, swap=True, insert=False, delete=False):
        self.txt_path = txt_path
        self.min_err = min_err
        self.max_err = max_err
        self.swap = swap
        self.insert = insert
        self.delete = delete
        self.text_mapper = TextMapper(chars)
        with open(self.txt_path) as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def get_batch(self, batch_size=64, max_length=100):
        n_err = np.random.randint(np.full(batch_size, self.max_err - self.min_err)) + self.min_err
        n_pos = np.random.randint(np.full((batch_size, self.max_err - self.min_err)), max_length) + self.min_err

        start_index = random.randint(0, len(self.lines) - batch_size)
        batch_lines = self.lines[start_index:batch_size]
        labels = torch.tensor(self.text_mapper.texts_to_labels(batch_lines))
        labels.scatter_(1, index, src)

        return

    def delete(self, text, delete_positions: List):


    def insert(self, text, insert_texts: List, insert_positions: List):
        r = []
        insert_positions = [0] + insert_positions
        for i_pos, insert_text in zip(range(1, len(insert_positions)), insert_texts):
            r += text[insert_positions[i_pos - 1]:insert_positions[i_pos]] + [insert_text]
        r += text[insert_positions[-1]:]
        return r


