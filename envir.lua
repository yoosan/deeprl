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

local envir = torch.class('deeprl.envir')

function envir:__init(config)
    self.win_height = config.win_height
    self.win_width = config.win_width
    self.state = nil
end

function envir:observe()
    local screen = torch.zeros(self.win_height, self.win_width)
    screen[{ self.state[1], self.state[2] }] = 1

    -- draw basket
    screen[{ self.win_height, self.state[3] - 1 }] = 2
    screen[{ self.win_height, self.state[3] }] = 2
    screen[{ self.win_height, self.state[3] + 1 }] = 2
    screen = screen:view(-1)
    return screen
end

function envir:reset()
    local init_col = math.random(1, self.win_width)
    local init_pos = math.random(2, self.win_width - 1)
    self.state = torch.Tensor({ 1, init_col, init_pos })
    return self:get_state()
end

function envir:get_state()
    local state = self.state
    return state[1], state[2], state[3]
end

function envir:get_reward()
    local row, col, pos = self:get_state()
    if row == self.win_height - 1 then -- reach the bottom
        if math.abs(col - pos) <= 1 then
            return 1 -- catch
        else
            return -1 -- not catch
        end
    else
        return 0
    end
end

function envir:game_over()
    if self.state[1] >= self.win_height - 1 then
        return true
    else
        return false
    end
end

function envir:update_state(action)
    local real_act = 0
    if action == 1 then
        real_act = -1
    elseif action == 2 then
        real_act = 0
    else
        real_act = 1
    end
    local row, col, pos = self:get_state()

    -- the basket moves one step, fruit falls one step
    local new_pos = math.min(self.win_width - 1, math.max(2, pos + real_act))
    row = row + 1
    self.state = torch.Tensor({ row, col, new_pos })
end

function envir:act(action)
    self:update_state(action)
    local reward = self:get_reward()
    local game_over = self:game_over()
    return self:observe(), reward, game_over, self:get_state()
end