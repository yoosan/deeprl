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

local screen = torch.class('deeprl.screen')

function screen:__init(config)
    self.win_height = config.win_height
    self.win_width = config.win_width
    self.screen = torch.Tensor(3, self.win_height, self.win_width):fill(0.8)
    self.win = nil
end

function screen:show(state)
    -- fill window with white color
    local row, col, pos = state[1], state[2], state[3]
    self.screen[{1, row, col}], self.screen[{2, row, col}], self.screen[{3, row, col}] = 0.5, 0.3, 0.2
    -- fill color for basket
    self.screen[{1, self.win_height, pos}] = 0.2
    self.screen[{2, self.win_height, pos}] = 0.3
    self.screen[{3, self.win_height, pos}] = 0.5
    self.screen[{1, self.win_height, pos-1}] = 0.2
    self.screen[{2, self.win_height, pos-1}] = 0.3
    self.screen[{3, self.win_height, pos-1}] = 0.5
    self.screen[{1, self.win_height, pos+1}] = 0.2
    self.screen[{2, self.win_height, pos+1}] = 0.3
    self.screen[{3, self.win_height, pos+1}] = 0.5
    self.win = image.display({image=self.screen, offscreen=false, win=self.win, zoom=30})
    self.screen:fill(0.8)
    -- sys.sleep(0.5)
end

function screen:show_car(state)
    local car_pos = state[1]
    self.screen[{1, self.win_height, car_pos}] = 0.5
    self.screen[{2, self.win_height, car_pos}] = 0.3
    self.screen[{3, self.win_height, car_pos}] = 0.2

    for i = 2, #state do
        local row = state[i][1]
        local col1, col2 = state[i][2], state[i][3]
        self.screen[{1, row, col1}] = 0.2
        self.screen[{2, row, col1}] = 0.3
        self.screen[{3, row, col1}] = 0.5
        self.screen[{1, row, col2}] = 0.2
        self.screen[{2, row, col2}] = 0.3
        self.screen[{3, row, col2}] = 0.5
    end

    self.win = image.display({image=self.screen, offscreen=false, win=self.win, zoom=30})
    self.screen:fill(0.8)
    -- sys.sleep(0.5)
end
