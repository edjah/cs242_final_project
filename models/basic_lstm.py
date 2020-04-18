import torch.nn as nn
from data_setup import TEXT, WORD_VECS


class BasicLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, emb_dropout=0.1, lstm_dropout=0.1, out_dropout=0.2):
        super().__init__()

        self.vocab_size = len(TEXT.vocab)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.out_dropout = nn.Dropout(p=out_dropout)
        self.embeddings = nn.Embedding.from_pretrained(WORD_VECS.clone(), freeze=False)

        self.lstm = nn.LSTM(
            WORD_VECS.shape[1], hidden_size, num_layers,
            dropout=lstm_dropout, batch_first=True
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_size, self.vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, hidden=None):
        """
        TODO: @Jason, is it wrong to ignore the hidden state here?
        """
        emb = self.emb_dropout(self.embeddings(x))
        output, hidden = self.lstm(emb)
        return self.out(self.out_dropout(output))
