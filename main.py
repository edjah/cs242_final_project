"""
Notes
-----
1.) Nenya: I tested it out and also checked the source code, and BPTTIterator
does not shuffle the batches between epochs. This may or may not be a problem.

2.) The embedding vectors are frozen. We might want to unfreeze them later.

3.) We should do some analysis to see what percentage of the the training
time is taken up by the RNN. This will give us upper bound on how fast we can go.
"""

import time
import torch

from data_setup import device, get_data_iterators
from models.basic_lstm import BasicLSTM


def evaluate(model, data_iterator):
    """
    Returns a tuple of (accuracy, perplexity)
    """
    model.eval()

    with torch.no_grad():
        loss_fn = torch.nn.NLLLoss(reduction='sum')
        loss = 0
        num_correct = 0
        total_words = 0

        for batch in data_iterator:
            log_probs = model.forward(batch.text)
            loss += loss_fn(log_probs.transpose(1, 2), batch.target)
            num_correct += (log_probs.argmax(dim=2) == batch.target).sum().float()
            total_words += batch.target.numel()

        return 100.0 * num_correct / total_words, torch.exp(loss / total_words)


def train_model(model, model_name, num_epochs=300, learning_rate=0.001,
                weight_decay=0, log_freq=1, batch_size=16):

    train_iter, val_iter, test_iter = get_data_iterators(batch_size)

    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = torch.nn.NLLLoss()

    train_start_time = time.time()
    best_val_ppl = float('inf')

    for epoch in range(num_epochs):
        try:
            model.train()
            ppl_ema = 0
            acc_ema = 0
            epoch_start_time = time.time()

            for i, batch in enumerate(train_iter):
                opt.zero_grad()

                log_probs = model.forward(batch.text)
                loss = loss_fn(log_probs.transpose(1, 2), batch.target)

                # compute gradients and update weights
                loss.backward()
                opt.step()

                # do some logging for perplexity and accuracy
                num_correct = (log_probs.detach().argmax(dim=2) == batch.target).sum().float()
                ppl = loss.detach().exp()
                acc = 100.0 * num_correct / batch.target.numel()

                ppl_ema = ppl_ema * 0.8 + ppl * 0.2
                acc_ema = acc_ema * 0.8 + acc * 0.2
                runtime = round(time.time() - epoch_start_time)
                print(f'\rEpoch {epoch} | Time: {runtime} sec | Batch #{i}/{len(train_iter)} | '
                      f'Train PPL: {ppl_ema:.2f} | Train Acc: {acc_ema:.2f}%', end='')

            print()

            # evaluate performance on the validation set
            if epoch == 0 or epoch == num_epochs - 1 or (epoch + 1) % log_freq == 0:
                val_acc, val_ppl = evaluate(model, val_iter)
                runtime = round(time.time() - train_start_time)
                print(f'Epoch {epoch} | Total Runtime: {runtime} sec | '
                      f'Val PPL: {val_ppl:.2f} | Val Acc: {val_acc:.2f}%\n')

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    torch.save(model.state_dict(), f'model_weights/{model_name}_best')

        except KeyboardInterrupt:
            print(f'\n\nStopped training after {epoch} epochs...')
            break

    print('\n\nFinal results\n=============')
    runtime = round(time.time() - train_start_time)
    train_acc, train_ppl = evaluate(model, train_iter)
    val_acc, val_ppl = evaluate(model, val_iter)
    test_acc, test_ppl = evaluate(model, test_iter)

    print(f'Train PPL: {train_ppl:<8.2f}    Train Accuracy: {train_acc:.2f}%')
    print(f'Val PPL:   {val_ppl:<8.2f}    Val Accuracy:   {val_acc:.2f}%')
    print(f'Test PPL:  {test_ppl:<8.2f}    Test Accuracy:  {test_acc:.2f}%')
    print(f'Total Runtime: {runtime} sec\n')


if __name__ == '__main__':
    # building the model
    model = BasicLSTM(
        hidden_size=1024, num_layers=1, emb_dropout=0.5,
        lstm_dropout=0.5, out_dropout=0.5,
    )
    model = model.to(device)

    # loading saved weights (comment this out if you want to start fresh)
    # model.load_state_dict(torch.load('model_weights/basic_lstm_best'))

    train_model(
        model, model_name='basic_lstm', num_epochs=20,
        learning_rate=0.001, weight_decay=0.0, log_freq=1,
        batch_size=16
    )
