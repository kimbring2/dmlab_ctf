local custom_observations = require 'decorators.custom_observations'
local game = require 'dmlab.system.game'
local game_types = require 'common.game_types'
local make_map = require 'common.make_map'
local maze_generation = require 'dmlab.system.maze_generation'
local map_maker = require 'dmlab.system.map_maker'
local random = require 'common.random'
local randomMap = random(map_maker:randomGen())
local setting_overrides = require 'decorators.setting_overrides'
local pickups = require 'common.pickups'
local maze_gen = require 'dmlab.system.maze_generation'
local debug_observations = require 'decorators.debug_observations'
local colors = require 'common.colors'
local events = require 'dmlab.system.events'
local image = require 'dmlab.system.image'
local color_bots = require 'common.color_bots'

local factory = {}


function factory.createLevelApi(kwargs)
  assert(kwargs.mapName)
  assert(kwargs.episodeLengthSeconds)
  assert(kwargs.botCount)
  local api = {}

  function api:gameType()
    return game_types.CAPTURE_THE_FLAG
  end

  local TEAMS = {'r', 'b'}
  function api:team(playerId, playerName)
    return TEAMS[playerId % 2 + 1]
  end

  function api:addBots()
    return color_bots:makeBots{
        count = kwargs.botCount,
        color = kwargs.color,
        skill = kwargs.skill,
    }
  end

  local characterSkinData
  local function characterSkins()
    if not characterSkinData then
      local playerDir = game:runFiles() .. '/baselab/game_scripts/player/'
      characterSkinData = {
          image.load(playerDir .. 'dm_character_skin_mask_a.png'),
          image.load(playerDir .. 'dm_character_skin_mask_b.png'),
          image.load(playerDir .. 'dm_character_skin_mask_c.png'),
      }
    end

    return characterSkinData
  end

  function api:playerModel(playerId, playerName)
    if playerId == 1 then
      return "crash_color"
    elseif playerId < 9 then
      return 'crash_color/skin' .. playerId - 1
    end

    return "crash"
  end

  local function updateSkin(playerSkin, rgbs)
    local skins = characterSkins()
    for i, charachterSkin in ipairs(skins) do
      local r, g, b = unpack(rgbs[i])
      local skinC = charachterSkin:clone()

      skinC:select(3, 1):mul(r / 255.0)
      skinC:select(3, 2):mul(g / 255.0)
      skinC:select(3, 3):mul(b / 255.0)
      playerSkin:cadd(skinC)
    end
  end

  local spawnCount = 1
  function api:updateSpawnVars(spawnVars)
    if spawnVars.classname == 'info_player_start' then
      -- Spawn facing East.
      spawnVars.angle = '0'
      spawnVars.randomAngleRange = '0'

      -- Make Bot spawn on first 'P' and player on second 'P'.
      spawnVars.nohumans = spawnCount > 1 and '1' or '0'
      spawnVars.nobots = spawnCount == 1 and '1' or '0'
      spawnCount = spawnCount + 1
    end

    return spawnVars
  end

  function api:modifyTexture(name, texture)
    if name == 'models/players/crash_color/skin_base1.tga' then
      texture:add{255, 0, 0, 0}
      return true
    end
    if name == 'models/players/crash_color/skin_base2.tga' then
      texture:add{0, 0, 255, 0}
      return true
    end

    return false
  end

  local function makeEntities(i, j, c, maker)
    if c == 'w' then
      return maker:makeEntity{
            i = i,
            j = j,
            classname = random:choice({
                'weapon_rocketlauncher',
                'weapon_lightning',
                'weapon_railgun',
            }),
        }
    elseif c == 'u' then
        return maker:makeEntity{
            i = i,
            j = j,
            classname = random:choice({
                'item_health_large',
                'item_armor_combat'
            }),
        }
    elseif c == 'p' then
        return maker:makeEntity{
            i = i,
            j = j,
            classname = random:choice({
                'powerup_redflag',
                'powerup_blueflag'
            }),
        }
    end
  end

  function api:nextMap()
    local MAP = [[
      *************************
      *           P           *
      *                       *
      *           w           *
      *                       *
      *           p           *
      *           p           *
      *                       *
      *           u           *
      *                       *
      *                       *
      *                       *
      *                       *
      *                       *
      *                       *
      *           P           *
      *************************
      ]]
    api._botColors = {}
    return kwargs.mapName
  end

  function api:newClientInfo(playerId, playerName, playerModel)
    local rgb
    if playerId == 1 or playerId == 3 or playerId == 5 then
      rgb = {colors.hsvToRgb(240.0, 1.0, 1.0)}
    elseif playerId == 2 or playerId == 4 or playerId == 6 then
      rgb = {colors.hsvToRgb(0.0, 1.0, 1.0)}
    else
      rgb = {colors.hsvToRgb(120.0 / 360, 1.0, 1.0)}
    end

    local _, _, id = string.find(playerModel, 'crash_color/skin(%d*)')
    id = id and id + 1 or 1
    api._botColors[id] = rgb
    events:add('newClientInfo' .. id)
  end

  local modifyTexture = api.modifyTexture
  function api:modifyTexture(name, texture)
    local _, _, id = string.find(name,
      'models/players/crash_color/skin_base(%d*).tga')
    if id then
      if id == '' then
        id = 1
      else
        id = tonumber(id) + 1
      end
      local rgb = api._botColors[id]
      updateSkin(texture, {rgb, rgb, rgb})
      events:add('skinModified' .. id)
      return true
    end
    return modifyTexture and modifyTexture(self, texture) or false
  end

  return api
end

return factory