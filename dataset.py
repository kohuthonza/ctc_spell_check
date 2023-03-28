import copy

import numpy as np
import random

from text_mapper import TextMapper
from text_mapper import chars


class SpellDataset:
    def __init__(self, txt_path, blank_symbol='�', pad_symbol='Ξ', replace_rnd=None, insert_rnd=None, delete_rnd=None):
        self.txt_path = txt_path
        self.blank_symbol = blank_symbol
        self.pad_symbol = pad_symbol
        self.index_blank_symbol = 0
        self.index_pad_symbol = 1
        self.replace_rnd = replace_rnd
        self.insert_rnd = insert_rnd
        self.delete_rnd = delete_rnd
        self.text_mapper = TextMapper(chars, joker=self.blank_symbol, pad=self.pad_symbol)
        self.rnd = self.replace_rnd is not None or self.insert_rnd is not None or self.delete_rnd is not None
        with open(self.txt_path) as f:
            self.lines = f.readlines()
        self.lines = [x.strip() for x in self.lines]
        self.labels = self.text_mapper.texts_to_labels(self.lines)
        self.labels = [list(x) for x in self.labels]
        self.lengths = [len(x) for x in self.labels]

    def __len__(self):
        return len(self.lines)

    def get_batch(self, batch_size=64):
        start_index = random.randint(0, len(self.lines) - batch_size)
        gt_lines = self.lines[start_index:start_index + batch_size]
        gt_labels = self.labels[start_index:start_index + batch_size]
        gt_lengths = self.lengths[start_index:start_index + batch_size]

        _target_labels = []
        target_lengths = []
        if self.rnd:
            for gt_label, gt_length in zip(gt_labels, gt_lengths):
                target_label = self.mutate_label(gt_label, gt_length)
                _target_labels.append(target_label)
                target_lengths.append(len(target_label))
        else:
            _target_labels = copy.deepcopy(gt_labels)
            target_lengths = copy.deepcopy(gt_lengths)

        target_labels = []
        pad_max_length = max(target_lengths)
        for target_label in _target_labels:
            target_labels.append(self._pad_label(target_label, pad_max_length))

        target_lines = self.text_mapper.labels_to_text(target_labels)

        return {'gt_lines': gt_lines,
                'target_lines': target_lines,
                'target_labels': target_labels,
                'target_lengths': target_lengths}

    def mutate_label(self, label, length):
        indexes = []
        inserts = []
        replaces = []
        if self.replace_rnd is not None:
            n_replace = self.replace_rnd()
            indexes += random.sample(range(length), n_replace)
            # skip blank and pad
            inserts += random.sample(range(2, len(self.text_mapper.chars)), n_replace)
            replaces += [1] * n_replace

        if self.delete_rnd is not None:
            n_delete = self.delete_rnd()
            indexes += random.sample(range(length), n_delete)
            inserts += [self.index_blank_symbol] * n_delete
            replaces += [1] * n_delete

        if self.insert_rnd:
            n_insert = self.insert_rnd()
            indexes += random.sample(range(length), n_insert)
            # skip blank and pad
            inserts += random.sample(range(2, len(self.text_mapper.chars)), n_insert)
            replaces += [0] * n_insert

        if indexes:

            indexes = [0] + indexes
            inserts = [self.index_blank_symbol] + inserts
            replaces = [0] + replaces

            edits = np.vstack((indexes, inserts, replaces))
            edits = edits[:, edits[0, :].argsort()]
            to_indexes = list(edits[0, 1:]) + [len(label)]

            return self._mutate_label(label, edits[0], to_indexes, edits[1], edits[2])
        else:
            return label

    def _mutate_label(self, label, from_indexes, to_indexes, inserts, replaces):
        o = []
        for from_index, to_index, insert, replace in zip(from_indexes, to_indexes, inserts, replaces):
            if insert == self.index_blank_symbol:
                o += label[from_index + replace:to_index]
            else:
                o += [insert] + label[from_index + replace:to_index]
        return o

    def _pad_label(self, label, max_length):
        padded_label = np.full(max_length, self.index_pad_symbol)
        label_length = len(label)
        left_pad = min(max_length - label_length, 5)
        padded_label[left_pad:left_pad + label_length] = label
        return padded_label


