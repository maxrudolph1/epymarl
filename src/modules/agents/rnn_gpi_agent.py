import torch.nn as nn
import torch.nn.functional as F
import torch

class RNNGPIAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNGPIAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim + args.policy_embed_size, args.n_actions)
        

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, policy_zs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        x = self.fc2(h)
        x = x.unsqueeze(1).expand(-1, policy_zs.shape[1], -1)
        x_z = torch.cat([x, policy_zs], dim=-1).view(-1, x.shape[-1] + policy_zs.shape[-1])

        q = self.fc3(x_z)
        q = q.view(inputs.shape[0], policy_zs.shape[1], -1)
        # print(q.shape)
        return q, h

