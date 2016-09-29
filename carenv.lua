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

local carenv = torch.class('deeprl.carenv')

function carenv:__init(config)
    self.win_height = config.win_height
    self.win_width = config.win_width
    self.gap = config.gap or 6
    self.state = nil
    self.heath = config.heath or 5
end

function carenv:observe()
    local screen = torch.zeros(self.win_height, self.win_width)
    local state = self:get_state()
    local car_pos = state[1]
    screen[{self.win_height, car_pos}] = 1
    for i = 2, #state do
        screen[{state[i][1], state[i][2]}] = 2
        screen[{state[i][1], state[i][3]}] = 2
    end
    screen = screen:view(-1)
    return screen
end

function carenv:reset()
    -- init the car position
    local init_pos = math.random(1, self.win_width)

    -- init barriers, 2 block per step
    local bar1_col = math.random(1, self.win_width)
    local bar2_col
    while true do
        bar2_col = math.random(1, self.win_width)
        if bar2_col ~= bar1_col then
            break
        end
    end
    self.state = {
        init_pos,
        {1, bar1_col, bar2_col},
    }
end

function carenv:get_state()
    return self.state
end

function carenv:get_reward()
    local state = self:get_state()
    local car_pos = state[1]
    local last_bar_row = state[#state][1]
    local last_bar1_col = state[#state][2]
    local last_bar2_col = state[#state][3]
    if last_bar_row == self.win_height then
        if last_bar1_col ~= car_pos and last_bar2_col ~= car_pos then
            -- avoid
            return 1
        else
            return -1
        end
    else
        return 0
    end
end

function carenv:game_over()
    local state = self.state
    local car_pos = state[1]
    local bar1 = state[#state][2]
    local bar2 = state[#state][3]
    local touch = bar1 == car_pos or bar2 == car_pos
    if state[#state][1] >= self.win_height and touch then
        return true
    else
        return false
    end
end

function carenv:update_state(action)
    local real_act = 0
    if action == 1 then
        real_act = -1
    elseif action == 2 then
        real_act = 0
    else
        real_act = 1
    end
    local state = self:get_state()
    local car_pos = state[1]
    local gen_bar1_col, gen_bar2_col

    for i = 2, #state do
        state[i][1] = state[i][1] + 1
    end

    if #state < math.floor(self.win_height/self.gap) + 2 then
        -- generate new barriers
        if (state[2][1] - 1) % self.gap == 0 then
            gen_bar1_col = math.random(1, self.win_width)
            while true do
                gen_bar2_col = math.random(1, self.win_width)
                if gen_bar2_col ~= gen_bar1_col then
                    break
                end
            end
            table.insert(state, 2, {1, gen_bar1_col, gen_bar2_col})
        end
    end
    if state[#state][1] > self.win_height then
        table.remove(state, #state)
    end
    -- the basket moves one step, fruit falls one step
    local new_pos = math.min(self.win_width, math.max(1, car_pos + real_act))
    state[1] = new_pos
    self.state = state
end

function carenv:act(action)
    self:update_state(action)
    local reward = self:get_reward()
    local game_over = self:game_over()
    return self:observe(), reward, game_over, self:get_state()
end