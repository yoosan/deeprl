--
--  Copyright (c) 2016, Horizon Robotics, Inc.
--  All rights reserved.
--
--  This source code is licensed under the MIT license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: Yao Zhou, yao.zhou@hobot.cc 
--  

local agent = torch.class('deeprl.agent')

function agent:__init(config)
    self.max_mem = config.max_mem
    self.policy_net = self:create_network()
    self.memory = {}
    self.criterion = nn.MSECriterion()
    self.bsize = config.bsize
    self.n_actions = config.n_actions
    self.n_states = config.n_states
    self.discount = config.discount
    self.hid_dim = config.hid_dim
    self.env = config.env
end

function agent:remember(mem_input)
    table.insert(self.memory, mem_input)
    if #self.memory > self.max_mem then
        table.remove(mem_input, 1)
    end
end

function agent:generate_batch()
    local mem_size = #self.memory
    local bsize = math.min(mem_size, self.bsize)
    local indices = torch.randperm(mem_size)
    -- inputs are screens, targets are actions
    local inputs = torch.zeros(bsize, self.n_states)
    local targets = torch.zeros(bsize, self.n_actions)

    for i = 1, bsize do
        local index = math.random(1, mem_size)
        local mem_input = self.memory[index]

        local target = self.policy_net:forward(mem_input.input_state):clone()

        local next_target = self.policy_net:forward(mem_input.next_state)
        local next_state_max_q = torch.max(next_target)
        if mem_input.game_over then
            target[mem_input.action] = mem_input.reward
        else
            -- reward + discount(gamma) * max_a' Q(s', a')
            -- expected Q-value of reward + gamma * max a' Q(s', a')
            target[mem_input.action] = mem_input.reward + self.discount * next_state_max_q
        end

        -- update the inputs and targets
        inputs[i] = mem_input.input_state
        targets[i] = target
    end
    return inputs, targets
end

function agent:create_network()
    local net = nn.Sequential()
    net:add(nn.Linear(self.n_states, self.hid_dim))
    net:add(nn.ReLU())
    net:add(nn.Linear(self.hid_dim, self.hid_dim))
    net:add(nn.ReLU())
    net:add(nn.Linear(self.hid_dim, self.n_actions))
    return net
end

function agent:train()
    
end