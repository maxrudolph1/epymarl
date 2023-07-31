from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np


# This multi-agent controller shares parameters between agents
class BasicMACGPI:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        # self.policy_mu = args.policy_mu
        # self.policy_sigma = args.policy_sigma
        if args.discrete_policies:
            assert self.args.num_policies == self.args.policy_embed_size

        self.num_policies = args.num_policies
        self.cur_policies = None
        self.policy_embed_size = args.policy_embed_size

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs

        if t_ep == 0:
            policy_zs = self.policy_choices(ep_batch, t_ep, t_env, bs, test_mode=test_mode)
        else:
            policy_zs = self.cur_policy_z
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        
        if self.args.discrete_gpi_policies <= 1:
            
            if not self.args.iterated_gpi:
                agent_outputs = self.forward(ep_batch, t_ep, policy_zs=policy_zs, test_mode=test_mode) # policy has not been added to batch yet, so need to pass in as argument
                chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
            else:
                ## TODO - add iterated GPI
                
                pass
        else:
            agent_outputs = self.discrete_policy_forward(ep_batch, t_ep, test_mode=test_mode) # policies are not indexed through input, but different model files
                                                                                              # find Qs for each discrete policy 
            chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        obs = ep_batch['actions']

        return chosen_actions

    def policy_choices(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Define a set of policy embeddings. Should be N_agents x N_policies x policy_embed_size
        # 
        if self.args.discrete_policies:
            policy_vecs = np.repeat(np.eye(self.num_policies)[np.newaxis, :,:], self.n_agents, axis=0)
        else:
            policy_vecs = np.repeat(np.random.random((self.num_policies,self.policy_embed_size))[np.newaxis, :,:], self.n_agents, axis=0)

        self.cur_policy_z = th.from_numpy(policy_vecs).float().to(self.args.device) # torch of the current policy vectors
        if ep_batch.batch_size == 1:
            self.cur_policy_z = self.cur_policy_z.unsqueeze(0)
        return self.cur_policy_z
    
    def get_chosen_policies(self):
        return self.cur_policy_z[np.newaxis, np.newaxis, :,:]
        
        
    def discrete_policy_forward(self, ep_batch, t, test_mode=False):

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs, next_hidden_states = [],[]

        for i in range(len(self.agents)):

            agent_out, hidden = self.agents[i](agent_inputs, self.hidden_states[i])
            agent_outs.append(agent_out.unsqueeze(2))
            next_hidden_states.append(hidden)
            
        agent_outs = th.cat(agent_outs, dim=2)
        self.hidden_states = next_hidden_states
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, self.args.discrete_gpi_policies, -1)
    
    def forward(self, ep_batch, t, policy_zs=None, test_mode=False, agent_id=0):
        
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if policy_zs is None:
            policy_zs = ep_batch["policy"][:, t] # 


        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, policy_zs) 

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, self.num_policies, -1)

    def init_hidden(self, batch_size):

        if self.args.discrete_gpi_policies <= 1:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        else:
            self.hidden_states = [agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1) for agent in self.agents]
            
    def parameters(self):
        if self.args.discrete_gpi_policies <= 1:
            return self.agent.parameters()
        else:
            return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        if self.args.discrete_gpi_policies > 1:
            for agent in self.agents:
                agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        if self.args.discrete_gpi_policies <= 1:
            self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        else:
            for pa, agent in zip(path, self.agents):
                agent.load_state_dict(th.load("{}/agent.th".format(pa), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        if self.args.discrete_gpi_policies <= 1:
            self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        else:
            self.agents = [agent_REGISTRY[self.args.agent](input_shape, self.args) for _ in range(len(self.args.checkpoint_path))]
            self.agent = self.agents[0]

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:

            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
