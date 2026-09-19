"""Minimap protocol, channel semantics and real-engine integration."""

import sys
from pathlib import Path

import numpy as np
import pytest

from fle.commons.observation.minimap import CHANNELS, MinimapCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from benchmark_tensor_obs import TensorClient  # noqa: E402


def test_minimap_decoder_scroll_invalidation_and_reset():
    c = MinimapCache()
    c.apply("M1,1,1,1,0,0,10,1,1,1;C1,0,0,10,0:1,128:200")
    assert c.values.shape == (14, 128, 128)
    assert c.values[0, 0, 0] == 1
    assert c.values[2, 0, 0] == 200
    assert c.values[13].sum() == 1
    c.apply("V1,0,0")
    assert c.values[2, 0, 0] == c.sampled_ticks[0, 0] == -1
    c.apply("S1,0,0,11")
    assert c.values[2, 0, 0] == 200
    assert c.sampled_ticks[0, 0] == 11
    # Moving origin into a previously clipped cached chunk preserves its data.
    c.apply("M1,1,1,1,-4,0,12,1,1,1")
    assert c.values[2, 0, 1] == 200
    assert c.values[13, 0, 1] == 1
    c.apply("M1,2,2,1,0,0,13,0,0,0;C1,0,0,99,128:900")
    assert np.all(c.values[:13] == -1)
    assert not c.values[13].any()
    c.apply("Moff,3;M1,2,1,1,0,0,14,0,0,0")
    assert not c.enabled
    with pytest.raises(ValueError, match="protocol"):
        c.apply("M2,4,1,1,0,0,14,0,0,0")


@pytest.fixture()
def minimap_world(tiered_rcon):
    rc = tiered_rcon
    setup = rc.send_command(
        """/sc game.tick_paused=true
obs_buildability_configure{enabled=false} obs_minimap_configure{enabled=false}
local s=game.surfaces["minimap-tests"] or game.create_surface("minimap-tests",{autoplace_controls={}})
s.request_to_generate_chunks({0,0},2) s.force_generate_chunk_requests()
for _,e in pairs(s.find_entities_filtered{area={{-32,-32},{64,64}}}) do e.destroy() end
local tiles={} for y=-32,63 do for x=-32,63 do tiles[#tiles+1]={name="grass-1",position={x,y}} end end s.set_tiles(tiles)
local f=game.forces["minimap-tests"] or game.create_force("minimap-tests") f.clear_chart(s)
storage.obs_char=s.create_entity{name="character",position={.5,12.5},force=f}
obs_minimap_configure{surface_index=s.index,force=f.name,center={x=0,y=0},chunk_budget=64,visibility="generated"} rcon.print("ready")"""
    )
    assert setup.strip() == "ready", setup
    c = MinimapCache()
    yield rc, c
    rc.send_command(
        "/sc obs_minimap_configure{enabled=false} if storage.obs_char and storage.obs_char.valid then storage.obs_char.destroy() end storage.obs_char=nil game.tick_paused=false game.speed=10"
    )


def drain(rc, c):
    response = rc.send_command("/sc obs_minimap_drain()") or ""
    assert response.startswith("M1"), response
    c.apply(response)
    return response


def fill(rc, c):
    for _ in range(20):
        drain(rc, c)
        if c.enabled and np.all(c.sampled_ticks >= 0):
            return
    pytest.fail("minimap did not fill")


def value(c, name, x, y):
    return c.values[
        CHANNELS.index(name), int((y - c.origin[1]) // 4), int((x - c.origin[0]) // 4)
    ]


def test_all_channels_and_generated_visibility(minimap_world):
    rc, c = minimap_world
    setup = rc.send_command(
        """/sc local s=game.surfaces["minimap-tests"] local f=game.forces["minimap-tests"]
local names={"iron-ore","copper-ore","coal","stone","uranium-ore","crude-oil"}
for i,n in ipairs(names) do assert(s.create_entity{name=n,position={(i-1)*4+.5,.5},amount=i*100}) end
s.set_tiles{{name="water",position={0,4}}}
assert(s.create_entity{name="tree-01",position={4.5,4.5}})
assert(s.create_entity{name="big-rock",position={8.5,4.5}})
assert(s.create_entity{name="iron-chest",position={12.5,4.5},force=f})
assert(s.create_entity{name="biter-spawner",position={20,12},force="enemy"})
assert(s.create_entity{name="small-biter",position={24.5,4.5},force="enemy"})
s.request_to_generate_chunks({160,160},1) s.force_generate_chunk_requests()
assert(s.create_entity{name="iron-ore",position={160.5,160.5},amount=999}) rcon.print("ready")"""
    )
    assert setup.strip() == "ready", setup
    fill(rc, c)
    assert value(c, "explored", 0, 0) == 1
    assert value(c, "explored", 160, 160) == 1
    assert value(c, "explored", -256, -256) == 0
    assert value(c, "iron_ore", 160, 160) == 999
    for i, name in enumerate(CHANNELS[2:8]):
        assert value(c, name, i * 4, 0) == (i + 1) * 100
    assert value(c, "water", 0, 4) == 1 / 16
    assert value(c, "trees", 4, 4) == 1
    assert value(c, "rocks_cliffs", 8, 4) > 0
    assert value(c, "friendly_structures", 12, 4) == 1 / 16
    assert value(c, "enemy_structures", 20, 12) > 0
    assert value(c, "enemy_units", 24, 4) == 1
    assert value(c, "player", 0.5, 12.5) == 1
    assert c.values[13].sum() == 1
    assert np.all(
        (c.values[[0, 1, 9, 10, 11]] >= 0) & (c.values[[0, 1, 9, 10, 11]] <= 1)
    )


def test_budget_events_and_silent_movement(minimap_world):
    rc, c = minimap_world
    fill(rc, c)
    rc.send_command(
        """/sc local s=game.surfaces["minimap-tests"]
storage.mm_chest=s.create_entity{name="iron-chest",position={12.5,4.5},force="minimap-tests",raise_built=true}
storage.mm_unit=s.create_entity{name="small-biter",position={24.5,4.5},force="enemy"}"""
    )
    response = drain(rc, c)
    assert "V" in response
    for _ in range(10):
        drain(rc, c)
    assert value(c, "friendly_structures", 12, 4) == 1 / 16
    assert value(c, "enemy_units", 24, 4) == 1
    rc.send_command(
        "/sc storage.mm_chest.destroy{raise_destroy=true} assert(storage.mm_unit.teleport({36.5,4.5})) game.forces['minimap-tests'].chart(game.surfaces['minimap-tests'],{{128,128},{159,159}})"
    )
    for _ in range(10):
        drain(rc, c)
    assert value(c, "friendly_structures", 12, 4) == 0
    assert value(c, "enemy_units", 24, 4) == 0
    assert value(c, "enemy_units", 36, 4) == 1
    rc.send_command(
        "/sc obs_minimap_configure{surface_index=game.surfaces['minimap-tests'].index,force='minimap-tests',center={x=0,y=0}}"
    )
    c = MinimapCache()
    drain(rc, c)
    assert int(rc.send_command("/sc rcon.print(storage.obs_minimap.last_samples)")) <= 4
    assert (c.sampled_ticks >= 0).sum() <= 4 * 64
    assert c.visibility == "generated"


def test_follow_surface_recovery_and_combined_client(minimap_world):
    rc, _ = minimap_world
    rc.send_command("/sc obs_minimap_configure{chunk_budget=64,visibility='generated'}")
    client = TensorClient()

    def poll():
        response = rc.send_command("/sc obs_all_drain()") or ""
        e, _, t = response.partition("~")
        client.apply_entity(e)
        client.apply_terrain(t)

    poll()
    c = client.minimap
    assert c.origin == (-256, -244)
    assert len(client.observation()) == 4
    assert client.observation(include_minimap=True)[4]["values"] is c.values
    assert len(client.observation(include_buildability=True, include_minimap=True)) == 6
    old_gen = c.generation
    rc.send_command("/sc storage.obs_char.teleport({8.5,12.5})")
    poll()
    assert c.origin == (-248, -244)
    assert c.generation == old_gen
    c.apply(rc.send_command("/sc obs_minimap_full_sync()"))
    assert c.generation > old_gen
    old_gen = c.generation
    rc.send_command("/sc storage.obs_char.teleport({0,0},game.surfaces[1])")
    poll()
    assert c.generation > old_gen
    assert c.surface == 1
    assert (c.sampled_ticks >= 0).sum() <= 64 * 64


def test_charted_visibility_does_not_leak_generated_terrain(minimap_world):
    rc, _ = minimap_world
    rc.send_command(
        "/sc local s=game.surfaces['minimap-tests'] local f=game.forces['minimap-tests'] "
        "f.clear_chart(s) f.cancel_charting(s) "
        "obs_minimap_configure{surface_index=s.index,force=f.name,center={x=0,y=0},chunk_budget=64,visibility='charted'}"
    )
    c = MinimapCache()
    fill(rc, c)
    assert c.visibility == "charted"
    assert np.all(c.values[:13] == 0)
    assert c.values[13].sum() == 1


def test_tile_event_and_cross_chunk_occupancy_union(minimap_world):
    rc, c = minimap_world
    fill(rc, c)
    response = rc.send_command("""/sc local s=game.surfaces['minimap-tests']
assert(s.create_entity{name='assembling-machine-1',position={31.5,20.5},force='minimap-tests',raise_built=true})
assert(s.create_entity{name='assembling-machine-1',position={31.5,20.5},force='minimap-tests',raise_built=true})
s.set_tiles({{name='water',position={-1,-1}}},true,false,false,true)
rcon.print('ready')""")
    assert response.strip() == "ready", response
    assert "V" in drain(rc, c)
    for _ in range(10):
        drain(rc, c)
    assert value(c, "water", -1, -1) == 1 / 16
    # A 3×3 occupied footprint straddles x=32 and y=20. Overlap is unioned.
    assert value(c, "friendly_structures", 28, 16) == 2 / 16
    assert value(c, "friendly_structures", 32, 16) == 1 / 16
    assert value(c, "friendly_structures", 28, 20) == 4 / 16
    assert value(c, "friendly_structures", 32, 20) == 2 / 16
