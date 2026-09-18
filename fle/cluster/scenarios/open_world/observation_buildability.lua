-- Progressive, bounded-work tile-level buildability cache. No work until enabled.
-- Binary values are exact engine queries at the transmitted sample tick, not a
-- guarantee about future ticks. Events invalidate immediately; silent changes
-- are covered by rolling refresh. Clients must retain sample ages.
local manifest = require("buildability_channels")
local M = {}
local SIDE, COUNT = 8, #manifest.channels
local dependencies

local function state() return storage.obs_buildability end
local function key(x, y) return x .. ":" .. y end

local function specs()
  if dependencies then return dependencies end
  dependencies = {}
  for i, spec in ipairs(manifest.channels) do
    local p = prototypes.entity[spec.name]
    assert(p, "Missing buildability prototype: " .. spec.name)
    local w, h = p.tile_width, p.tile_height
    if spec.direction == 4 or spec.direction == 12 then w, h = h, w end
    local fluid = false
    for _, name in ipairs(spec.members) do
      if #prototypes.entity[name].fluidbox_prototypes > 0 then fluid = true end
    end
    dependencies[i] = {x = (w % 2) * .5, y = (h % 2) * .5, fluid = fluid}
  end
  return dependencies
end

local function reorder(s, x, y)
  s.x0 = math.floor(x / SIDE + .5) * SIDE - s.size / 2
  s.y0 = math.floor(y / SIDE + .5) * SIDE - s.size / 2
  s.order = {}
  local keep = {}
  for cy = s.y0 / SIDE, (s.y0 + s.size) / SIDE - 1 do
    for cx = s.x0 / SIDE, (s.x0 + s.size) / SIDE - 1 do
      local k = key(cx, cy)
      keep[k] = true
      s.order[#s.order + 1] = {x = cx, y = cy, key = k,
        distance = (cx * SIDE + 4 - x)^2 + (cy * SIDE + 4 - y)^2}
    end
  end
  table.sort(s.order, function(a, b)
    if a.distance ~= b.distance then return a.distance < b.distance end
    if a.y ~= b.y then return a.y < b.y end
    return a.x < b.x
  end)
  for k in pairs(s.cache) do if not keep[k] then s.cache[k] = nil end end
  s.cursor, s.near_cursor, s.metadata = 1, 1, true
  -- Existing queued invalidations may refer to evicted blocks; drain skips them.
end

function M.configure(options)
  local o = options or {}
  local old = state()
  local generation = old and old.generation + 1 or 1
  if o.enabled == false then
    storage.obs_buildability = {enabled = false, generation = generation, metadata = true}
    return
  end
  assert(script.active_mods.base == "2.0.73", "Buildability manifest requires Factorio base 2.0.73")
  for name in pairs(script.active_mods) do
    assert(name == "base" or name == "core", "Buildability classes must be regenerated for mod: " .. name)
  end
  local size = o.size or 128
  local budget = o.check_budget or 256
  local refresh = o.refresh_ticks or 600
  assert(size >= 16 and size <= 288 and size % 16 == 0, "size must be a multiple of 16 in [16,288]")
  assert(budget >= 128 and budget <= 8192 and budget % 128 == 0, "check_budget must be a multiple of 128 in [128,8192]")
  assert(refresh >= 1 and refresh % 1 == 0, "refresh_ticks must be a positive integer")
  local surface = game.surfaces[o.surface_index or 1]
  local force = game.forces[o.force or "player"]
  assert(surface and force, "Invalid buildability surface/force")
  specs()
  local s = {enabled = true, generation = generation, size = size, budget = budget,
    refresh = refresh, surface = surface.index, force = force.index,
    follow = o.center == nil, cache = {}, order = {}, epochs = {}, observed = {},
    pending_channels = {}, regions = {}, queue = {}, queued = {}, head = 1, tail = 0}
  for i = 1, COUNT do s.epochs[i] = 0 end
  storage.obs_buildability = s
  local p = o.center or {x = 0, y = 0}
  reorder(s, p.x, p.y)
end

function M.invalidate_all()
  local s = state()
  if not (s and s.enabled) then return end
  s.generation = s.generation + 1
  s.cache, s.regions, s.pending_channels = {}, {}, {}
  s.queue, s.queued, s.head, s.tail = {}, {}, 1, 0
  s.cursor, s.metadata = 1, true
end

local function enqueue(s, cx, cy, channel)
  local k = key(cx, cy) .. ":" .. channel
  if s.queued[k] then return end
  s.tail = s.tail + 1
  s.queue[s.tail] = {x = cx, y = cy, channel = channel, key = k}
  s.queued[k] = true
end

function M.invalidate_area(surface_index, area)
  local s = state()
  if not (s and s.enabled and surface_index == s.surface) then return end
  -- 16 tiles covers supported footprints, mining range, and underground reach.
  local x0 = math.max(s.x0 / SIDE, math.floor((area.left_top.x - 16) / SIDE))
  local y0 = math.max(s.y0 / SIDE, math.floor((area.left_top.y - 16) / SIDE))
  local x1 = math.min((s.x0 + s.size) / SIDE - 1, math.floor((area.right_bottom.x + 16) / SIDE))
  local y1 = math.min((s.y0 + s.size) / SIDE - 1, math.floor((area.right_bottom.y + 16) / SIDE))
  if x0 > x1 or y0 > y1 then return end
  s.regions[#s.regions + 1] = {x0, y0, x1, y1}
  if #s.regions > 64 then M.invalidate_all() return end
  for cy = y0, y1 do for cx = x0, x1 do
    local block = s.cache[key(cx, cy)]
    if block then for channel, entry in pairs(block) do
      entry.dirty = true
      enqueue(s, cx, cy, channel)
    end end
  end end
end

function M.invalidate_fluids()
  local s = state()
  if not (s and s.enabled) then return end
  -- A changed connection can merge distant fluid systems: local invalidation
  -- alone is unsound. Invalidate fluid-dependent channels across the viewport.
  for i, d in ipairs(specs()) do if d.fluid and not s.pending_channels[i] then
    s.epochs[i] = s.epochs[i] + 1
    s.pending_channels[i] = true
  end end
  s.cursor = 1
end

local function signature(e)
  local p, b = e.position, e.bounding_box
  local out = {e.name, p.x, p.y, e.direction, e.force.index,
    b.left_top.x, b.left_top.y, b.right_bottom.x, b.right_bottom.y}
  if #e.prototype.fluidbox_prototypes > 0 then
    for i=1,#e.fluidbox do
      local f = e.fluidbox[i]
      out[#out+1] = f and f.name or "-"
    end
    local ok, recipe = pcall(function() return e.get_recipe() end)
    if ok and recipe then out[#out+1] = recipe.name end
  end
  return table.concat(out, ":")
end

function M.invalidate_entity(e)
  local s = state()
  if not (s and s.enabled and e and e.valid and e.surface.index == s.surface) then return end
  if e.unit_number then
    local old = s.observed[e.unit_number]
    if old then M.invalidate_area(e.surface.index, old.area) end
    s.observed[e.unit_number] = {signature = signature(e), area = e.bounding_box}
  end
  M.invalidate_area(e.surface.index, e.bounding_box)
  if #e.prototype.fluidbox_prototypes > 0 then M.invalidate_fluids() end
end

function M.reconcile_entity(e)
  local s = state()
  if not (s and s.enabled and e and e.valid and e.unit_number and e.surface.index == s.surface) then return end
  local old = s.observed[e.unit_number]
  if not old or old.signature ~= signature(e) then M.invalidate_entity(e) end
end

function M.forget(unit_number)
  local s = state()
  if s and s.enabled and unit_number then s.observed[unit_number] = nil end
end

function M.tiles(event)
  local s = state()
  if not (s and s.enabled) then return end
  local surface = event.surface_index or (event.surface and event.surface.index)
  if surface ~= s.surface then return end
  local area
  for _, tile in pairs(event.tiles or {}) do
    local p = tile.position
    if p then
      if not area then area = {left_top={x=p.x,y=p.y},right_bottom={x=p.x+1,y=p.y+1}}
      else
        area.left_top.x, area.left_top.y = math.min(area.left_top.x,p.x), math.min(area.left_top.y,p.y)
        area.right_bottom.x, area.right_bottom.y = math.max(area.right_bottom.x,p.x+1), math.max(area.right_bottom.y,p.y+1)
      end
    end
  end
  if area then M.invalidate_area(surface, area) M.invalidate_fluids() end
end

function M.full_sync()
  local s = state()
  if not (s and s.enabled) then return end
  -- Replay valid cached blocks progressively, without recomputing them.
  s.generation = s.generation + 1
  s.cursor, s.metadata = 1, true
end

local function compute(s, job)
  local spec, offset = manifest.channels[job.channel], specs()[job.channel]
  local surface, lo, hi, checks = game.surfaces[s.surface], 0, 0, 0
  for n = 0, 63 do
    local x, y = job.x * SIDE + n % SIDE, job.y * SIDE + math.floor(n / SIDE)
    local params = {name = spec.name, direction = spec.direction, force = s.force,
      position = {x + offset.x, y + offset.y}, build_check_type = defines.build_check_type.manual}
    local ok = surface.can_place_entity(params)
    checks = checks + 1
    if ok then
      params.build_check_type = defines.build_check_type.ghost_revive
      ok = surface.can_place_entity(params)
      checks = checks + 1
    end
    if ok then
      if n < 32 then lo = bit32.bor(lo, bit32.lshift(1, n))
      else hi = bit32.bor(hi, bit32.lshift(1, n - 32)) end
    end
  end
  return {lo = lo, hi = hi, tick = game.tick, epoch = s.epochs[job.channel]}, checks
end

function M.drain()
  local s = state()
  if not s then return "" end
  if not s.enabled then
    if not s.metadata then return "" end
    s.metadata = false
    return "Boff," .. s.generation
  end
  local char = storage.obs_char
  if s.follow and char and char.valid then
    if char.surface.index ~= s.surface or char.force.index ~= s.force then
      s.surface, s.force = char.surface.index, char.force.index
      M.invalidate_all()
      s.player_x, s.player_y = nil, nil
    end
    local p = char.position
    if s.player_x and (p.x ~= s.player_x or p.y ~= s.player_y) then
      M.invalidate_area(s.surface, {left_top={x=s.player_x,y=s.player_y},right_bottom={x=s.player_x,y=s.player_y}})
      M.invalidate_entity(char)
    end
    s.player_x, s.player_y = p.x, p.y
    if math.abs(p.x - (s.x0 + s.size / 2)) > 24 or math.abs(p.y - (s.y0 + s.size / 2)) > 24 then
      reorder(s, p.x, p.y)
    end
  end
  local out = {}
  if s.metadata then
    out[#out+1] = table.concat({"B" .. manifest.id, s.generation, s.surface, s.force, s.x0, s.y0, s.size}, ",")
    s.metadata = false
  end
  for _, r in ipairs(s.regions) do
    out[#out+1] = table.concat({"R" .. s.generation, r[1], r[2], r[3], r[4]}, ",")
  end
  s.regions = {}
  for channel in pairs(s.pending_channels) do
    out[#out+1] = "J" .. s.generation .. "," .. (channel - 1)
  end
  s.pending_channels = {}
  local quota, completed, scans = s.budget / 128, 0, 0
  s.last_checks = 0
  while completed < quota and scans < quota * 32 do
    scans = scans + 1
    local job
    -- Reserve alternate jobs for the normal traversal to avoid dirty starvation.
    s.turn = (s.turn or 0) + 1
    if s.turn % 2 == 1 and s.head <= s.tail then
      job = s.queue[s.head]
      s.queue[s.head], s.queued[job.key] = nil, nil
      s.head = s.head + 1
    elseif s.turn % 2 == 1 then
      -- Refresh the nearest nine blocks while the far field is still filling.
      local near = (s.near_cursor or 1) - 1
      local block = s.order[math.floor(near / COUNT) + 1]
      local channel = near % COUNT + 1
      s.near_cursor = (near + 1) % (math.min(9, #s.order) * COUNT) + 1
      local cached = s.cache[block.key]
      local entry = cached and cached[channel]
      if entry and (entry.dirty or entry.epoch ~= s.epochs[channel] or game.tick - entry.tick >= s.refresh) then
        job = {x = block.x, y = block.y, channel = channel}
      end
    end
    if not job then
      local index = s.cursor - 1
      local block = s.order[math.floor(index / COUNT) + 1]
      job = {x = block.x, y = block.y, channel = index % COUNT + 1}
      s.cursor = s.cursor % (#s.order * COUNT) + 1
    end
    if job.x >= s.x0 / SIDE and job.x < (s.x0 + s.size) / SIDE and
        job.y >= s.y0 / SIDE and job.y < (s.y0 + s.size) / SIDE then
      local k = key(job.x, job.y)
      local block = s.cache[k]
      local entry = block and block[job.channel]
      if not entry or entry.dirty or entry.epoch ~= s.epochs[job.channel] or game.tick - entry.tick >= s.refresh then
        local checks
        entry, checks = compute(s, job)
        s.last_checks = s.last_checks + checks
        if not block then block = {} s.cache[k] = block end
        block[job.channel] = entry
        entry.sent = nil
      end
      if entry.sent ~= s.generation then
        out[#out+1] = table.concat({"b" .. s.generation, job.channel - 1, job.x, job.y,
          entry.tick, string.format("%08x%08x", entry.lo, entry.hi)}, ",")
        entry.sent = s.generation
        completed = completed + 1
      end
    end
  end
  if s.head > s.tail then s.queue, s.queued, s.head, s.tail = {}, {}, 1, 0 end
  return table.concat(out, ";")
end

return M
