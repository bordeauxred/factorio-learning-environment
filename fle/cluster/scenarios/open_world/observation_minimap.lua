-- Independent progressive 512-tile minimap: 128 cells, four tiles per cell.
-- Chunk queries are bounded in count, not in entity density or wall-clock time.
local M = {}
local WATER = {"water", "deepwater", "water-green", "deepwater-green"}
local RESOURCES = {["iron-ore"]=3,["copper-ore"]=4,coal=5,stone=6,["uranium-ore"]=7,["crude-oil"]=8}
local STRUCTURES = {}
for _, t in ipairs({"accumulator","agricultural-tower","ammo-turret","arithmetic-combinator",
  "artillery-turret","assembling-machine","asteroid-collector","beacon","boiler",
  "burner-generator","cargo-bay","constant-combinator","container","curved-rail-a",
  "curved-rail-b","decider-combinator","electric-energy-interface","electric-pole",
  "electric-turret","fluid-turret","furnace","fusion-generator","fusion-reactor","gate",
  "generator","half-diagonal-rail","heat-interface","heat-pipe","infinity-container",
  "infinity-pipe","inserter","lab","lamp","land-mine","legacy-curved-rail",
  "legacy-straight-rail","lightning-attractor","linked-belt","linked-container",
  "loader","loader-1x1","logistic-container","mining-drill","offshore-pump","pipe",
  "pipe-to-ground","power-switch","programmable-speaker","pump","radar","rail-chain-signal",
  "rail-ramp","rail-signal","rail-support","reactor","roboport","rocket-silo",
  "selector-combinator","simple-entity-with-owner","simple-entity-with-force","solar-panel",
  "space-platform-hub","splitter","storage-tank","straight-rail","thruster","train-stop",
  "transport-belt","underground-belt","wall"}) do STRUCTURES[t] = true end
local function state() return storage.obs_minimap end
local function key(x,y) return x .. ":" .. y end
local function inside(s,cx,cy)
  return cx*32 < s.x0+512 and cx*32+32 > s.x0 and cy*32 < s.y0+512 and cy*32+32 > s.y0
end
local function reorder(s,x,y)
  local old_next = s.order and s.order[s.cursor or 1]
  s.x0, s.y0 = math.floor(x/4)*4-256, math.floor(y/4)*4-256
  s.order = {}
  for cy=math.floor(s.y0/32),math.floor((s.y0+511)/32) do
    for cx=math.floor(s.x0/32),math.floor((s.x0+511)/32) do
      s.order[#s.order+1]={x=cx,y=cy,key=key(cx,cy),d=(cx*32+16-x)^2+(cy*32+16-y)^2}
    end
  end
  table.sort(s.order,function(a,b) if a.d~=b.d then return a.d<b.d end if a.y~=b.y then return a.y<b.y end return a.x<b.x end)
  s.cursor=1
  for i,c in ipairs(s.order) do if old_next and c.key==old_next.key then s.cursor=i end end
  for k,c in pairs(s.cache) do if not inside(s,c.x,c.y) then s.cache[k]=nil end end
end
function M.configure(o)
  o=o or {}
  local old=state()
  local gen=old and old.generation+1 or 1
  if o.enabled==false then storage.obs_minimap={enabled=false,generation=gen,disabled_pending=true} return end
  local budget=o.chunk_budget or 4
  assert(budget%1==0 and budget>=1 and budget<=64,"chunk_budget must be an integer in [1,64]")
  local visibility=o.visibility or "generated"
  assert(visibility=="charted" or visibility=="generated", "visibility must be charted or generated")
  local surface=game.surfaces[o.surface_index or 1]
  local force=game.forces[o.force or "player"]
  assert(surface and force,"Invalid minimap surface/force")
  local s={enabled=true,generation=gen,budget=budget,surface=surface.index,force=force.index,
    follow=o.center==nil,center=o.center,visibility=visibility,cache={},queue={},queued={},pending={},head=1,tail=0,turn=0}
  storage.obs_minimap=s
  local p=o.center or {x=0,y=0}
  reorder(s,p.x,p.y)
end
function M.invalidate_all()
  local s=state() if not(s and s.enabled) then return end
  s.generation=s.generation+1
  s.cache,s.queue,s.queued,s.pending={},{},{},{}
  s.head,s.tail,s.cursor=1,0,1
end
function M.full_sync()
  local s=state() if not(s and s.enabled) then return end
  s.generation=s.generation+1
  for _,c in pairs(s.cache) do c.sent=nil end
  s.cursor=1
end
function M.invalidate_area(surface,area)
  local s=state() if not(s and s.enabled and surface==s.surface) then return end
  for cy=math.max(math.floor(area.left_top.y/32),math.floor(s.y0/32)),math.min(math.floor(area.right_bottom.y/32),math.floor((s.y0+511)/32)) do
    for cx=math.max(math.floor(area.left_top.x/32),math.floor(s.x0/32)),math.min(math.floor(area.right_bottom.x/32),math.floor((s.x0+511)/32)) do
      local k=key(cx,cy)
      s.pending[k]={x=cx,y=cy}
      if not s.queued[k] then
        s.tail=s.tail+1 s.queue[s.tail]={x=cx,y=cy,key=k} s.queued[k]=true
      end
    end
  end
end
function M.entity(e)
  local s=state() if not(s and s.enabled) then return end
  if e and e.valid then M.invalidate_area(e.surface.index,e.bounding_box) end
end
function M.tiles(event)
  local s=state() if not(s and s.enabled) then return end
  local surface=event.surface_index or (event.surface and event.surface.index)
  if surface~=s.surface then return end
  local area
  for _,t in pairs(event.tiles or {}) do
    local p=t.position
    if not area then area={left_top={x=p.x,y=p.y},right_bottom={x=p.x+1,y=p.y+1}}
    else
      area.left_top.x,area.left_top.y=math.min(area.left_top.x,p.x),math.min(area.left_top.y,p.y)
      area.right_bottom.x,area.right_bottom.y=math.max(area.right_bottom.x,p.x+1),math.max(area.right_bottom.y,p.y+1)
    end
  end
  if area then M.invalidate_area(surface,area) end
end
function M.chart(event)
  local s=state()
  if s and s.enabled and event.force.index==s.force then
    local p=event.position
    M.invalidate_area(event.surface_index,{left_top={x=p.x*32,y=p.y*32},right_bottom={x=p.x*32+31,y=p.y*32+31}})
  end
end
local function sample(s,c)
  local surface,force=game.surfaces[s.surface],game.forces[s.force]
  local values={}
  for i=1,13*64 do values[i]=0 end
  local x0,y0=c.x*32,c.y*32
  -- Generated matches FLE terrain visibility; charted optionally enforces game exploration.
  if surface.is_chunk_generated({c.x,c.y}) and (s.visibility=="generated" or force.is_chunk_charted(surface,{c.x,c.y})) then
    for i=1,64 do values[i]=1 end
    local function cell(p)
      local x,y=math.floor((p.x-x0)/4),math.floor((p.y-y0)/4)
      if x>=0 and x<8 and y>=0 and y<8 then return y*8+x+1 end
    end
    local function add(channel,p,v)
      local i=cell(p) if i then i=(channel-1)*64+i values[i]=values[i]+v end
    end
    local area={{x0,y0},{x0+32,y0+32}}
    for _,t in ipairs(surface.find_tiles_filtered{area=area,name=WATER}) do add(2,t.position,1/16) end
    local occupancy={}
    local function occupy(channel,e)
      local b=e.bounding_box
      for y=math.max(y0,math.floor(b.left_top.y)),math.min(y0+31,math.ceil(b.right_bottom.y)-1) do
        for x=math.max(x0,math.floor(b.left_top.x)),math.min(x0+31,math.ceil(b.right_bottom.x)-1) do
          local i=(channel-1)*64+math.floor((y-y0)/4)*8+math.floor((x-x0)/4)+1
          local bit=bit32.lshift(1,(y%4)*4+x%4)
          occupancy[i]=bit32.bor(occupancy[i] or 0,bit)
        end
      end
    end
    for _,e in ipairs(surface.find_entities_filtered{area=area}) do
      if e.type=="resource" then
        local channel=RESOURCES[e.name] if channel then add(channel,e.position,e.amount) end
      elseif e.type=="tree" then add(9,e.position,1)
      elseif e.type=="simple-entity" or e.type=="cliff" then occupy(10,e)
      elseif e.force.name=="enemy" then
        if e.type=="unit" then add(13,e.position,1)
        elseif e.type=="unit-spawner" or e.type=="turret" then occupy(12,e) end
      elseif STRUCTURES[e.type] and (e.force==force or force.get_friend(e.force)) then occupy(11,e) end
    end
    for i,bits in pairs(occupancy) do
      local n=0 for j=0,15 do if bit32.band(bits,bit32.lshift(1,j))~=0 then n=n+1 end end
      values[i]=n/16
    end
  end
  local sparse={}
  for i,v in ipairs(values) do if v~=0 then sparse[#sparse+1]=(i-1)..":"..v end end
  return table.concat(sparse,",")
end
function M.drain()
  local s=state() if not s then return "" end
  if not s.enabled then
    if not s.disabled_pending then return "" end
    s.disabled_pending=false return "Moff,"..s.generation
  end
  local char=storage.obs_char
  local px,py,present=0,0,0
  if char and char.valid then
    if s.follow and (s.surface~=char.surface.index or s.force~=char.force.index) then
      s.surface,s.force=char.surface.index,char.force.index M.invalidate_all()
    end
    if char.surface.index==s.surface then px,py,present=char.position.x,char.position.y,1 end
  end
  local center=s.center or {x=px,y=py}
  if math.floor(center.x/4)*4-256~=s.x0 or math.floor(center.y/4)*4-256~=s.y0 then reorder(s,center.x,center.y) end
  local out={table.concat({"M1",s.generation,s.surface,s.force,s.x0,s.y0,game.tick,px,py,present,s.visibility},",")}
  for _,c in pairs(s.pending) do out[#out+1]="V"..s.generation..","..c.x..","..c.y end
  s.pending={}
  s.last_samples=0
  local visited={}
  for _=1,s.budget do
    s.turn=s.turn+1
    local c
    if s.turn%2==1 and s.head<=s.tail then
      c=s.queue[s.head] s.queue[s.head]=nil s.queued[c.key]=nil s.head=s.head+1
    else c=s.order[s.cursor] s.cursor=s.cursor%#s.order+1 end
    if inside(s,c.x,c.y) and not visited[c.key] then
      visited[c.key]=true
      local payload=sample(s,c) s.last_samples=s.last_samples+1
      local old=s.cache[c.key]
      if not old or old.payload~=payload or old.sent~=s.generation then
        out[#out+1]="C"..s.generation..","..c.x..","..c.y..","..game.tick..","..payload
      else out[#out+1]="S"..s.generation..","..c.x..","..c.y..","..game.tick end
      s.cache[c.key]={x=c.x,y=c.y,payload=payload,sent=s.generation}
    end
  end
  if s.head>s.tail then s.queue,s.queued,s.head,s.tail={},{},1,0 end
  return table.concat(out,";")
end
return M
