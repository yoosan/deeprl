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

local learner = torch.class('deeprl.learner')

function learner:__init(config)
    self.task = config.task
    self.epoch = config.epoch
    self.env_config = config.env_config
    self.agent_config = config.agent_config
    self.agent = deeprl.agent(self.agent_config)
    if config.task == 'car' then
        self.envir = deeprl.carenv(self.env_config)
    else
        self.envir = deeprl.envir(self.env_config)
    end
    self.epsilon = config.epsilon
    self.screen = deeprl.screen(config.env_config)
end

function learner:run()
    local score = 0
    for i = 1, self.epoch do
        -- init environment
        local error = 0
        self.envir:reset()
        local game_over = false

        -- init state
        local cur_state = self.envir:observe()

        while game_over ~= true do
            local action
            if math.randf() <= self.epsilon then
                action = math.random(1, self.agent_config.n_actions)
            else
                -- forward
                local q = self.agent.policy_net:forward(cur_state)
                local max, idx = torch.max(q, 1)
                action = idx[1]
            end

            if self.epsilon > 0.001 then
                self.epsilon = self.epsilon * 0.999
            end

            local next_state, reward, go = self.envir:act(action)
            game_over = go
            if reward == 1 then score = score + 100 end

            self.agent:remember({
                input_state = cur_state,
                action = action,
                reward = reward,
                next_state = next_state,
                game_over = game_over,
            })

            -- self.screen:show(self.envir.state)
            cur_state = next_state

            -- batch training
            local inputs, targets = self.agent:generate_batch()
            error = error + self.agent:train(inputs, targets)
        end
        collectgarbage()
        print(string.format('Epoch %d : error = %f : Score %d', i, error, score))
    end
end

function learner:test(steps)
    local score = 0
    for i = 1, steps do
        local game_over = false
        local row, col, pos = self.envir:reset()
        local cur_state = self.envir:observe()
        while not game_over do
            local q = self.agent.policy_net:forward(cur_state)
            local max, idx = torch.max(q, 1)
            local action = idx[1]
            local next_state, reward, go, row, col, pos = self.envir:act(action)
            cur_state = next_state
            game_over = go
            reward = reward > 0 and 1 or 0
            score = score + reward * 100
            self.screen:show(self.envir.state)
        end
        print(string.format('step %d, score is %d', i, score))
    end
    os.exit()
end

function learner:run_car()
    for i = 1, self.epoch do
        -- init environment
        local error = 0
        local score = 0
        self.envir:reset()
        local game_over = false

        -- init state
        local cur_state = self.envir:observe()
        while game_over ~= true do
            local action
            if math.randf() <= self.epsilon then
                action = math.random(1, self.agent_config.n_actions)
            else
                -- forward
                local q = self.agent.policy_net:forward(cur_state)
                local max, idx = torch.max(q, 1)
                action = idx[1]
            end

            if self.epsilon > 0.001 then
                self.epsilon = self.epsilon * 0.999
            end

            local next_state, reward, go = self.envir:act(action)
            game_over = go
            if reward == 1 then score = score + 100 end

            self.agent:remember({
                input_state = cur_state,
                action = action,
                reward = reward,
                next_state = next_state,
                game_over = game_over,
            })

            -- self.screen:show(self.envir.state)
            cur_state = next_state

            -- batch training
            local inputs, targets = self.agent:generate_batch()
            error = error + self.agent:train(inputs, targets)
        end
        collectgarbage()
        print(string.format('Epoch %d : error = %f : Score %d', i, error, score))
    end
end

function learner:test_car(steps)
    for i = 1, steps do
        local score = 0
        local game_over = false
        self.envir:reset()
        local cur_state = self.envir:observe()
        while not game_over do
            local q = self.agent.policy_net:forward(cur_state)
            local max, idx = torch.max(q, 1)
            local action = idx[1]
            local next_state, reward, go, positions = self.envir:act(action)
            cur_state = next_state
            game_over = go
            reward = reward > 0 and 1 or 0
            score = score + reward * 100
            self.screen:show_car(positions)
        end
        collectgarbage()
        print(string.format('step %d, score is %d', i, score))
    end
    os.exit()
end