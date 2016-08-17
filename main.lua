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

deeprl = {}
require('torch')
require('nn')
require('optim')
require('agent.lua')
require('env.lua')
require('learner.lua')
require('utils.lua')

local cmd = torch.CmdLine()

cmd:text('params setting')
cmd:option('-seed', os.time(), 'initial random seed')
cmd:option('-win_height', 18, 'environment window height')
cmd:option('-win_width', 12, 'environment window width')
cmd:option('-win_width', 12, 'environment window width')
cmd:option('-max_men', 1000, 'max memory size')
cmd:option('-bsize', 16, 'training batch size')
cmd:option('-n_actions', 3, 'number of actions')
cmd:option('-discount', 0.9, 'discount factor gamma ')
cmd:option('-hid_dim', 128, 'dimension of hidden states')
cmd:text()

-- parse arguments
local opt = cmd:parse(arg or {})
math.randomseed(opt.seed)

local env_config = {
    win_height = opt.win_height,
    win_width = opt.win_width,
}

local agent_config = {
    max_len = opt.max_len,
    bsize = opt.bsize,
    n_actions = opt.n_actions,
    n_states = opt.win_height * opt.win_width,
    discount = opt.discount,
    hid_dim = opt.hid_dim,
    optim_config = {
        learningRate = 0.1,
    }
}

local env = deeprl.env(env_config)
local agent = deeprl.agent(agent_config)

print(env)
print(agent)