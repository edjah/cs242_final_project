import torch
import torchtext

import time
from collections import defaultdict


if torch.cuda.is_available():
    print('GPU is available and will be used.')
    device = torch.device('cuda:0')
else:
    print('GPU is not available, so CPU will be used.')
    device = torch.device('cpu')


TEXT = torchtext.data.Field(batch_first=True)
train, val, test = torchtext.datasets.PennTreebank.splits(TEXT)
TEXT.build_vocab(train, vectors=torchtext.vocab.GloVe(name='840B', dim=300))

WORD_VECS = TEXT.vocab.vectors


def get_data_iterators(batch_size=16):
    # NOTE: this shuffle=True does NOT work right now. We should probably make
    # a custom Iterator class which does actually do the shuffling correctly
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        datasets=(train, val, test), bptt_len=35, batch_size=batch_size,
        device=device, shuffle=True,
    )
    return train_iter, val_iter, test_iter


class Profiler:
    times = defaultdict(float)
    counts = defaultdict(int)

    def __init__(self, key):
        self.key = key

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        dur = time.time() - self.start_time
        Profiler.times[self.key] += dur
        Profiler.counts[self.key] += 1

    @classmethod
    def reset(cls):
        cls.times.clear()
        cls.counts.clear()

    @classmethod
    def print_times(cls):
        for k, v in sorted(cls.times.items(), key=lambda x: -x[1]):
            avg_t = v / cls.counts[k]
            print(f'key: {k} | tot time: {v:.6f} sec | count: {cls.counts[k]} | avg time: {avg_t:.6f} sec')
