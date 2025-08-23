import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class DKTplus_dataset(Dataset):
    def __init__(self, q_sequences, r_sequences, t_sequences):
        self.q_sequences = q_sequences
        self.r_sequences = r_sequences
        self.t_sequences = t_sequences

    def __len__(self):
        return len(self.r_sequences)

    def __getitem__(self, index):
        if len(self.q_sequences[index]) < 2:
            return None

        q, r, t = (
            self.q_sequences[index][:-1],
            self.r_sequences[index][:-1],
            self.t_sequences[index][:-1],
        )
        q_next, r_next = self.q_sequences[index][1:], self.r_sequences[index][1:]

        return (
            torch.LongTensor(q),
            torch.LongTensor(r),
            torch.FloatTensor(t),
            torch.LongTensor(q_next),
            torch.FloatTensor(r_next),
        )


def DKTplus_collate(batch):
    batch = [item for item in batch if item is not None]

    if not batch:
        return None, None, None, None, None, None

    q, r, t, q_next, r_next = zip(*batch)

    q, r, t = (
        pad_sequence(q, batch_first=True),
        pad_sequence(r, batch_first=True),
        pad_sequence(t, batch_first=True),
    )

    q_next, r_next = (
        pad_sequence(q_next, batch_first=True),
        pad_sequence(r_next, batch_first=True),
    )

    sequence_mask = q_next > 0

    return q, r, t, q_next, r_next, sequence_mask
