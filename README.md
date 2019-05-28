# Correcting-romanian-language-sentences

General:
- Explored the idea of directly correcting the sentences using the character encoding.
- Tested only on a dataset of 12 000 sequences (good & wrong sequences)
- character level encoding of sentences.
- padded sequences to the same length.
- added start and end of sequence tokens.
- loss is more important, since the sequences are padded to the same max length.
- hot-encoded the target sequences.
- Used an Embedding() layer for input sequences (tested only with 128 output dimension of the dense embedding).
- Used 99% of data as training, and 1% as validation.

On model_1 -> 30 epochs -> loss: 0.2755 - acc: 0.9231 - val_loss: 0.2992 - val_acc: 0.9212.
On model 4 -> 10 epochs -> loss: 0.3694 - categorical_accuracy: 0.8886 - val_loss: 0.3923 - val_categorical_accuracy: 0.8830
