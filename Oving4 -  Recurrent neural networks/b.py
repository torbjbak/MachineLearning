import torch
import torch.nn as nn
import numpy as np

np.set_printoptions(precision=2)

class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a'
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'c'
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'f'
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h'
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'l'
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'm'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'n'
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'o'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'p'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'r'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 's'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 't'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # '🎩'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # '🐀'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # '🐈'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # '🏢'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # '🧑'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # '🧢'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]   # '👶'
]
encoding_size = len(char_encodings)

index_to_char = [' ', 'a', 'c', 'f', 'h', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', '🎩', '🐀', '🐈', '🏢', '🧑', '🧢', '👶']

x_train = torch.tensor([
    [[char_encodings[4]], [char_encodings[1]], [char_encodings[12]], [char_encodings[0]]],  # 'hat '
    [[char_encodings[10]], [char_encodings[1]], [char_encodings[12]], [char_encodings[0]]], # 'rat '
    [[char_encodings[2]], [char_encodings[1]], [char_encodings[12]], [char_encodings[0]]],  # 'cat '
    [[char_encodings[3]], [char_encodings[5]], [char_encodings[1]], [char_encodings[12]]],  # 'flat'
    [[char_encodings[6]], [char_encodings[1]], [char_encodings[12]], [char_encodings[12]]], # 'matt'
    [[char_encodings[2]], [char_encodings[1]], [char_encodings[9]], [char_encodings[0]]],   # 'cap '
    [[char_encodings[11]], [char_encodings[8]], [char_encodings[7]], [char_encodings[0]]]   # 'son '
])

y_train = torch.tensor([
    [char_encodings[13], char_encodings[13], char_encodings[13], char_encodings[13]], # '🎩'
    [char_encodings[14], char_encodings[14], char_encodings[14], char_encodings[14]], # '🐀'
    [char_encodings[15], char_encodings[15], char_encodings[15], char_encodings[15]], # '🐈'
    [char_encodings[16], char_encodings[16], char_encodings[16], char_encodings[16]], # '🏢'
    [char_encodings[17], char_encodings[17], char_encodings[17], char_encodings[17]], # '🧑'
    [char_encodings[18], char_encodings[18], char_encodings[18], char_encodings[18]], # '🧢'
    [char_encodings[19], char_encodings[19], char_encodings[19], char_encodings[19]]  # '👶'
])

model = LongShortTermMemoryModel(encoding_size)

text = ['hat', 'rat', 'cat', 'flat', 'matt', 'cap', 'son', 'rt ', 'rats', 'ht', 'ct', 'ft', 'mt', 'cp', 'sn', 'tar']

optimizer = torch.optim.RMSprop(model.parameters(), 0.0005)
for epoch in range(500):
    for i in range(7):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if epoch%10 == 9:
        print_out = ""
        for i in range(len(text)):
            y = 0
            model.reset()
            for j in range(len(text[i])):
                y = model.f(torch.tensor([[char_encodings[index_to_char.index(text[i][j])]]]))

            print_out += text[i] + ":" + index_to_char[y.argmax(1)] + " | "

        print(print_out)


