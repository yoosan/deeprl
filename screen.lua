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
end

function screen:show(state)
    -- fill window with white color
    local scr = torch.Tensor(3, self.win_height, self.win_width):fill(0.8)
    local w
    local row, col, pos = state[1], state[2], state[3]
    scr[{1, row, col}], scr[{2, row, col}], scr[{3, row, col}] = 0.5, 0.3, 0.2
    local basket = scr:sub(1, 3, self.win_height, self.win_height, pos - 1, pos + 1)
    -- fill color
    basket[{1,}], basket[{2, }], basket[{3, }] = 0.2, 0.3, 0.5
    w = image.display({image=scr, offscreen=false, win=w, zoom=30})
    scr:fill(0.8)
--    sys.sleep(0.5)
end
