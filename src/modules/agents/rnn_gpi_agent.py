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
        self.fc3 = nn.Linear(args.hidden_dim + args.policy_hidden_dim , args.n_actions)
        
        self.policy_fc = nn.Linear(args.num_policies, args.policy_hidden_dim)
        

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, policy_zs):
        print(inputs.shape)
        x = F.relu(self.fc1(inputs))
        print(x.shape)
        # print(hidden_state.shape)
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        print(h_in.shape)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        x = self.fc2(h)
        x = x.unsqueeze(2).repeat(1, 1, self.args.num_policies, 1) # repeat tensor for all policies (bs, n_agents, n_policies, hidden_dim)
        x = x.view(-1, x.shape[-1]) # (bs * n_agents * n_policies, hidden_dim)
        
        policy_zs = policy_zs.reshape(-1, policy_zs.shape[-1]) # reshape policy from (bs, n_agents, n_policies, policy_dim) to (bs * n_agents * n_policies, policy_dim)
        z_embed = F.relu(self.policy_fc(policy_zs))
        if x.shape[0] != z_embed.shape[0]: # added for fixed batch running
            z_embed = z_embed.repeat(self.args.batch_size_run, 1)
        # print(x.shape, z_embed.shape)

        x = torch.cat([x, z_embed], dim=-1)

        q = self.fc3(x)
        q = q.view(-1, self.args.num_policies, self.args.n_actions)
        print()
        return q, h

