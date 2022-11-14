
from utils import *
from torch import optim

'''Learner file (BC + DAgger)'''

class custom_dataset:
	def __init__(self, data, targets):
		self.data = data; self.targets = targets

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		current_sample = self.data[idx, :]; current_target = self.targets[idx]; current_sample = torch.from_numpy(current_sample).to(torch.float32); current_target = torch.from_numpy(current_target).to(torch.long)
		return(current_sample, current_target)

class BC:
    def __init__(self, net, loss_fn):
        self.net = net
        self.loss_fn = loss_fn
        
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)

    def training(self, states, actions, env):
        # Dataloader
        dataset = custom_dataset(data = states, targets = actions)
        dataload = torch.utils.data.DataLoader(dataset, batch_size = 16, drop_last = True, shuffle=False, num_workers = 0)
        number_of_epochs = 50

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(number_of_epochs):
            for i, (x, y) in enumerate(dataload):
                # Initials
                x = x.to(device); y = y.to(device)
                # Forward Pass
                ypred = (self.net(x)).to(torch.float32); y = torch.squeeze(y)
                # Backward Pass
                self.opt.zero_grad()
                loss = self.loss_fn(ypred, y); loss.backward(); self.opt.step()
                # count = count+1

        return self.net

    def learn(self, env, states, actions, n_steps=1e4, truncate=True):
        # TODO: Implement this method. Return the final greedy policy (argmax_policy).
        self.net = self.training(states, actions, env)

        return self.net
    
class DAgger:
    # CONSTRUCTOR
    def __init__(self, net, loss_fn, expert):
        self.net = net
        self.loss_fn = loss_fn
        self.expert = expert
        
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)

    # HELPER FUNCTION
    def training(self, states, actions, env):
        # Dataloader
        dataset = custom_dataset(data = states, targets = actions)
        dataload = torch.utils.data.DataLoader(dataset, batch_size = 16, drop_last = True, shuffle=True, num_workers = 0)
        number_of_epochs = 50

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(number_of_epochs):
            for i, (x, y) in enumerate(dataload):
                # Initials
                x = x.to(device); y = y.to(device)
                # Forward Pass
                ypred = (self.net(x)).to(torch.float32); y = torch.squeeze(y)
                # Backward Pass
                self.opt.zero_grad()
                loss = self.loss_fn(ypred, y); loss.backward(); self.opt.step()

        return self.net

    # HELPER FUNCTION
    def trajectories(self, number_of_trajectories, function, network):
        states, actions = function(network)
        states = np.array(states); actions = np.array(actions)
        if number_of_trajectories>1:
            for i in range(number_of_trajectories-1):
                s, a = function(network)
                states = np.concatenate((states,np.array(s)), axis=0); actions = np.concatenate((actions,np.array(a)), axis=0)
        return states, actions

    # HELPER FUNCTION
    def iteration_0_action_mod(self, learner_actions_i0):
        learner_actions_i0_mod = np.empty(len(learner_actions_i0))
        for i in range(0, learner_actions_i0_mod.shape[0]):
            learner_actions_i0_mod[i] = np.argmax(learner_actions_i0[i])
        return learner_actions_i0_mod

    # LEARNER FUNCTION
    def learn(self, env, n_steps=1e4, truncate=False):
        # TODO: Implement this method. Return the final greedy policy (argmax_policy).
        # Make sure you are making the learning process fundamentally expert-interactive

        # --- Part I ---
        function = lambda network: rollout(network, env, False)
        number_of_trajectories = 1
        learner_states_i0, learner_actions_i0 = self.trajectories(number_of_trajectories, function, self.net)
        learner_actions_i0_mod = self.iteration_0_action_mod(learner_actions_i0[:,None])

        states = learner_states_i0; actions = learner_actions_i0_mod
        self.net = self.training(states, actions[:,None], env)

        # --- Part II ---
        number_of_iterations = 15
        for i in range(number_of_iterations):
            print('DAgger Iteration', i+1)

            learner_states, _ = self.trajectories(number_of_trajectories, function, self.net); expert_actions, _ = self.expert.predict(learner_states)

            states = np.concatenate((states, learner_states), axis=0); actions = np.concatenate((actions, expert_actions), axis=0)
            
            self.net = self.training(states, actions[:,None], env)
        
        return self.net
