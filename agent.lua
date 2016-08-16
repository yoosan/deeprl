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
    self.network = self:create_network()
    self.memory = {}
    self.criterion = nn.MSECriterion()
    self.bsize = config.bsize
    self.n_actions = config.n_actions
    self.n_states = config.n_states
end

function agent:remember(mem_input)
    table.insert(self.memory, mem_input)
    if #self.memory > self.max_mem then
        table.remove(mem_input, 1)
    end
end

function agent:gen_batch()
    local mem_size = #self.memory
    local bsize = math.min(mem_size, self.bsize)
    -- inputs are screens, targets are actions
    local inputs = torch.zeros(bsize, self.n_states)
    local targets = torch.zeros(bsize, self.n_actions)

end