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
    self.memory = {}
    self.max_mem = config.max_mem or 1000
    self.duel = config.duel or true
    self.bsize = config.bsize
    self.n_actions = config.n_actions
    self.n_states = config.n_states
    self.discount = config.discount
    self.hid_dim = config.hid_dim
    self.policy_net = self:create_network()
    self.criterion = nn.MSECriterion()
    self.params, self.grad_params = self.policy_net:getParameters()
    self.optim_config = config.optim_config or {
        learningRate = 0.1,
    }
end

function agent:remember(mem_input)
    table.insert(self.memory, mem_input)
    if #(self.memory) > self.max_mem then
        table.remove(mem_input, 1)
    end
end

function agent:create_network()
    local net = nn.Sequential()
    net:add(nn.Linear(self.n_states, self.hid_dim))
    net:add(nn.ReLU())
    net:add(nn.Linear(self.hid_dim, self.hid_dim))
    net:add(nn.ReLU())
    net:add(nn.Linear(self.hid_dim, self.n_actions))
    if self.duel then
        return self:create_duel_network()
    else
        return net
    end
end

function agent:create_duel_network()

    local DuelAggregator = function(m)
        local aggregator = nn.Sequential()
        local aggParallel = nn.ParallelTable()

        -- Advantage duplicator (for calculating and subtracting mean)
        local advDuplicator = nn.Sequential()
        local advConcat = nn.ConcatTable()
        advConcat:add(nn.Identity())
        -- Advantage mean duplicator
        local advMeanDuplicator = nn.Sequential()
        advMeanDuplicator:add(nn.Mean(1, 1))
        advMeanDuplicator:add(nn.Replicate(m, 2, 2))
        advConcat:add(advMeanDuplicator)
        advDuplicator:add(advConcat)
        -- Subtract mean from advantage values
        advDuplicator:add(nn.CSubTable())

        -- Add value and advantage duplicators
        aggParallel:add(nn.Replicate(m, 2, 2))
        aggParallel:add(advDuplicator)

        -- Calculate Q^ = V^ + A^
        aggregator:add(aggParallel)
        aggregator:add(nn.CAddTable())

        return aggregator
    end

    local net = nn.Sequential()
    net:add(nn.Linear(self.n_states, self.hid_dim))
    net:add(nn.ReLU())

    local valStream = nn.Sequential()
    valStream:add(nn.Linear(self.hid_dim, self.hid_dim))
    valStream:add(nn.ReLU(true))
    valStream:add(nn.Linear(self.hid_dim, 1)) -- Predicts value for state

    -- Advantage approximator A^(s, a)
    local advStream = nn.Sequential()
    advStream:add(nn.Linear(self.hid_dim, self.hid_dim))
    advStream:add(nn.ReLU(true))
    -- Predicts action-conditional advantage
    advStream:add(nn.Linear(self.hid_dim, self.n_actions))

    -- Streams container
    local streams = nn.ConcatTable()
    streams:add(valStream)
    streams:add(advStream)

    -- Add dueling streams
    net:add(streams)
    net:add(DuelAggregator(self.n_actions))
    net:add(nn.View(-1))
    return net
end

function agent:generate_batch()
    local mem_size = #self.memory
    local bsize = math.min(mem_size, self.bsize)

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

function agent:train(inputs, targets)
    local loss = 0
    local function feval(x)
        self.grad_params:zero()
        local preds = self.policy_net:forward(inputs)
        loss = self.criterion:forward(preds, targets)
        local grad_output = self.criterion:backward(preds, targets)
        self.policy_net:backward(inputs, grad_output)
        return loss, self.grad_params
    end

    local _, fs = optim.adagrad(feval, self.params, self.optim_config)
    loss = loss + fs[1]
    return loss
end