import torch
import torchtext

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
