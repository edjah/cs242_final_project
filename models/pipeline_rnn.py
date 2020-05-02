import torch
import torch.nn as nn
from data_setup import TEXT, WORD_VECS, Profiler

from collections import deque


# TODO: do I need the @custom_fwd decorator???

class BatchedRnnTanhCell(torch.autograd.Function):
    result = None
    idx = 0

    @staticmethod
    def reset():
        BatchedRnnTanhCell.result = None
        BatchedRnnTanhCell.idx = 0

    @staticmethod
    def forward(ctx, weight_ih, weight_hh, bias_ih, bias_hh, input, hidden):
        result = BatchedRnnTanhCell.result[BatchedRnnTanhCell.idx]
        BatchedRnnTanhCell.idx += 1
        ctx.save_for_backward(result, weight_ih, weight_hh, bias_ih, bias_hh, input, hidden)
        return result

    @staticmethod
    def backward(ctx, grad):
        """
        grad_tanh.shape = torch.Size([3])
        input.shape = torch.Size([5])
        weight_ih.shape = torch.Size([3, 5])
        weight_hh.shape = torch.Size([3, 3])
        hidden.shape = torch.Size([3])
        """
        result, weight_ih, weight_hh, bias_ih, bias_hh, input, hidden = ctx.saved_tensors

        # first compute the grad w.r.t the tanh op
        grad_tanh = grad * (1 - result * result)

        # now compute the grad w.r.t the matmuls
        grad_weight_ih = grad_tanh.t() @ input
        grad_input = grad_tanh @ weight_ih

        grad_weight_hh = grad_tanh.t() @ hidden
        grad_hidden = grad_tanh @ weight_hh

        # now compute the grad w.r.t the biases
        grad_bias_ih = grad_tanh
        grad_bias_hh = grad_tanh

        return grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh, grad_input, grad_hidden

    @staticmethod
    def batch_rnn_cell(weight_ih, weight_hh, bias_ih, bias_hh, *args):
        BatchedRnnTanhCell.reset()

        inputs = args[:len(args)//2]
        hiddens = args[len(args)//2:]

        with torch.no_grad():
            stacked_inputs = torch.stack(inputs)
            stacked_hiddens = torch.stack(hiddens)
            result = (
                stacked_inputs @ weight_ih.t() + bias_ih +
                stacked_hiddens @ weight_hh.t() + bias_hh
            )
            BatchedRnnTanhCell.result = torch.tanh(result).unbind()

        return tuple(
            BatchedRnnTanhCell.apply(weight_ih, weight_hh, bias_ih, bias_hh, i, h)
            for i, h in zip(inputs, hiddens)
        )


class PipelineRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        self.rnn = nn.RNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity)

        self.sentences = []
        self.targets = []
        self.hiddens = []
        self.outputs = []

        self.finished = deque()

    def forward(self, x):
        # assume batch_first=True, so x.shape = (batch_size, seq_len, emb_size)
        output = []
        hidden = torch.zeros(x.shape[0], self.rnn.hidden_size, device=x.device)
        for i in range(x.shape[1]):
            hidden = self.rnn.forward(x[:, i], hidden)
            output.append(hidden)

        # shape of output is (seq_len, batch_size, hidden_size)
        # need to transpose it to (batch_size, seq_len, hidden_size)
        return torch.stack(output).transpose(0, 1)

    def add_to_pipeline(self, sentence, target):
        """
        Adds a `sentence` and it's corresponding `target` to the pipeline
        `sentence.shape` should be (batch_size, seq_len, emb_size)
        """
        with Profiler('rnn_add_to_pipeline'):
            # store the sentences in reverse order, so that we can just pop from the end
            self.sentences.append(list(sentence.unbind(1))[::-1])
            self.targets.append(target)
            self.hiddens.append(torch.zeros(sentence.shape[0], self.rnn.hidden_size, device=sentence.device))
            self.outputs.append([])

    def pipeline_step(self):
        """
        Performs one step of the pipeline operation

        Returns:
            - If a result has completed: a tuple of (rnn_output, target).
              The rnn_output will be of shape (batch_size, seqlen, hidden_size)

            - Otherwise: None
        """
        with Profiler('rnn_pipeline_step'):
            if len(self.finished) > 0:
                out, y = self.finished.popleft()
                return out, y

            if len(self.sentences) == 0:
                raise StopIteration

            with Profiler('rnn_pipeline_input_building'):
                inputs = [s.pop() for s in self.sentences]

            with Profiler('rnn_pipeline_forward_and_update'):

                with Profiler('rnn_pipeline_clone'):
                    weight_ih = self.rnn.weight_ih.clone()
                    weight_hh = self.rnn.weight_hh.clone()
                    bias_ih = self.rnn.bias_ih.clone()
                    bias_hh = self.rnn.bias_hh.clone()

                with Profiler('rnn_pipeline_batch_rnn_cell'):
                    output_hiddens = BatchedRnnTanhCell.batch_rnn_cell(
                        weight_ih, weight_hh, bias_ih, bias_hh,
                        *inputs, *self.hiddens
                    )

                with Profiler('rnn_pipeline_update'):
                    self.hiddens = output_hiddens
                    for output, h in zip(self.outputs, output_hiddens):
                        output.append(h)

            with Profiler('rnn_finished_check'):
                state = zip(self.sentences, self.targets, self.hiddens, self.outputs)

                self.sentences = []
                self.targets = []
                self.hiddens = []
                self.outputs = []

                for s, t, h, o in state:
                    if len(s) == 0:
                        # need to stack the outputs along dim=1 which is the hidden
                        # dimension. dim=0 is the batch dimension
                        self.finished.append((torch.stack(o, dim=1), t))
                    else:
                        self.sentences.append(s)
                        self.targets.append(t)
                        self.hiddens.append(h)
                        self.outputs.append(o)

            if len(self.finished) > 0:
                out, y = self.finished.popleft()
                return out, y


class PipelineModel(nn.Module):
    def __init__(self, hidden_size, num_layers, emb_dropout=0.1, lstm_dropout=0.1, out_dropout=0.2):
        super().__init__()

        self.vocab_size = len(TEXT.vocab)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.out_dropout = nn.Dropout(p=out_dropout)
        self.embeddings = nn.Embedding.from_pretrained(WORD_VECS.clone(), freeze=False)

        self.rnn = PipelineRNN(WORD_VECS.shape[1], hidden_size)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, self.vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        emb = self.emb_dropout(self.embeddings(x))
        output = self.rnn.forward(emb)
        return self.out(self.out_dropout(output))

    def update_grad(self):
        self.rnn.update_grad()

    def pipeline(self, data_iterator):
        for batch in data_iterator:
            with Profiler('model_pipeline_emb'):
                emb = self.emb_dropout(self.embeddings(batch.text))

            self.rnn.add_to_pipeline(emb, batch.target)
            res = self.rnn.pipeline_step()

            if res is not None:
                with Profiler('model_pipeline_linear_out'):
                    rnn_output, y = res
                    result = self.out(self.out_dropout(rnn_output))
                yield result, y

        # wait for the the remaining elements in the batch to finish
        # pipeline_step() will raise StopIteration when there's nothing
        # left to process
        while True:
            try:
                res = self.rnn.pipeline_step()
                if res is not None:
                    with Profiler('model_pipeline_linear_out'):
                        rnn_output, y = res
                        result = self.out(self.out_dropout(rnn_output))
                    yield result, y
            except StopIteration:
                return
