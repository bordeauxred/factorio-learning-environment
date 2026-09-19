"""Progressive cache protocol, engine parity and invalidation regression tests."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from fle.commons.observation.buildability import (
    BuildabilityCache,
    CHANNEL_COUNT,
    CHANNEL_FOR,
    CHANNELS,
    MANIFEST,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from benchmark_tensor_obs import TensorClient  # noqa: E402


def meta(generation=1, x=0, y=0, size=16):
    return f"B{MANIFEST['id']},{generation},1,1,{x},{y},{size}"


def test_manifest_and_tile_decoding():
    assert CHANNEL_COUNT == 63
    assert len(CHANNEL_FOR) == 79 * 4
    c = BuildabilityCache()
    c.apply_record(meta())
    assert np.all(c.values == -1)
    c.apply_record("b1,0,0,0,12,0000000180000000")
    assert c.values[0, 0, 0] == c.values[0, 7, 7] == 1
    assert (c.values == 1).sum() == 2
    assert (c.values >= 0).sum() == 64
    # Only fresh, known-blocked cells are forbidden: the sampled block holds
    # 64 cells of which 2 are legal, and nothing outside it is ever masked.
    blocked = c.certainly_blocked_mask(15, 3)
    assert blocked.sum() == 62
    assert not blocked[c.values != 0].any()
    assert not c.certainly_blocked_mask(16, 3).any()  # stale -> allowed
    assert not c.certainly_blocked_mask(11, 3).any()  # negative age -> allowed
    known, age = c.freshness(15, age_scale=60)
    assert known.sum() == 64 and age[known == 0].min() == 1.0
    assert not c.apply_record("t0:0")
    lua = (
        Path(__file__).parents[2]
        / "fle/cluster/scenarios/open_world/buildability_channels.lua"
    ).read_text()
    assert MANIFEST["id"] in lua
    for channel in CHANNELS:
        assert f'name = "{channel["name"]}", direction = {channel["direction"]}' in lua


def test_recenter_invalidation_and_old_generation():
    c = BuildabilityCache()
    c.apply_record(meta())
    c.apply_record("b1,0,1,0,12,ffffffffffffffff")
    c.apply_record(meta(x=8))
    assert np.all(c.values[0, :8, :8] == 1)
    assert np.all(c.values[:, :, 8:] == -1)
    c.apply_record("R1,1,0,1,0")
    assert np.all(c.values == -1)
    c.apply_record("b1,1,1,0,13,ffffffffffffffff")
    c.apply_record("J1,1")
    assert np.all(c.values == -1)
    c.apply_record(meta(generation=2))
    c.apply_record("b1,0,0,0,999,ffffffffffffffff")
    assert np.all(c.values == -1)
    c.apply_record("Boff,3")
    assert c.size == 0
    c.apply_record(meta(generation=2))
    assert c.size == 0
    with pytest.raises(ValueError, match="manifest"):
        c.apply_record("Bwrong,4,1,1,0,0,16")


@pytest.fixture()
def world(tiered_rcon):
    rc = tiered_rcon
    rc.send_command(
        """/sc game.tick_paused=true
local s=game.surfaces[1] s.request_to_generate_chunks({2048,2048},2) s.force_generate_chunk_requests()
for _,e in pairs(s.find_entities_filtered{area={{2016,2016},{2080,2080}}}) do e.destroy{raise_destroy=true} end
local tiles={} for y=2016,2079 do for x=2016,2079 do tiles[#tiles+1]={name="grass-1",position={x,y}} end end
s.set_tiles(tiles) obs_buildability_configure{size=16,check_budget=8192,center={x=2048,y=2048}}"""
    )
    c = TensorClient()
    yield rc, c
    rc.send_command(
        "/sc obs_buildability_configure{enabled=false} game.tick_paused=false game.speed=10"
    )


def poll(rc, c):
    response = rc.send_command("/sc obs_all_drain()") or ""
    entities, _, terrain = response.partition("~")
    c.apply_entity(entities)
    c.apply_terrain(terrain)
    return terrain


def fill(rc, c, limit=12):
    for _ in range(limit):
        poll(rc, c)
        if c.buildability.size and np.all(c.buildability.values >= 0):
            return
    pytest.fail("cache did not finish the bounded small-window sweep")


def test_default_observation_window_is_progressive_and_bounded(world):
    rc, c = world
    rc.send_command(
        "/sc obs_buildability_configure{check_budget=256,center={x=2048,y=2048}}"
    )
    response = poll(rc, c)
    assert c.buildability.values.shape == (63, 128, 128)
    assert c.buildability.sampled_ticks.shape == (63, 16, 16)
    assert (c.buildability.values >= 0).sum() == 2 * 64
    assert (
        int(rc.send_command("/sc rcon.print(storage.obs_buildability.last_checks)"))
        <= 256
    )
    assert sum(r.startswith("b") for r in response.split(";")) == 2
    ys, xs = np.where((c.buildability.values >= 0).any(axis=0))
    assert np.max(abs(xs - 64)) <= 8
    assert np.max(abs(ys - 64)) <= 8
    assert len(c.observation()) == 4
    build = c.observation(include_buildability=True)[4]
    assert build["values"] is c.buildability.values
    assert build["origin"] == c.buildability.origin
    assert build["tick"] == c.tick


def test_small_window_matches_all_engine_representatives(world):
    rc, c = world
    fill(rc, c)
    representatives = (
        "{"
        + ",".join(
            '{name="%s",direction=%d}' % (ch["name"], ch["direction"])
            for ch in CHANNELS
        )
        + "}"
    )
    response = rc.send_command(
        """/sc local s=storage.obs_buildability local out={}
for _,a in ipairs("""
        + representatives
        + """) do
 local p=prototypes.entity[a.name] local w,h=p.tile_width,p.tile_height
 if a.direction==4 or a.direction==12 then w,h=h,w end
 local bits={} for y=s.y0,s.y0+s.size-1 do for x=s.x0,s.x0+s.size-1 do
 local q={name=a.name,direction=a.direction,position={x+(w%2)*.5,y+(h%2)*.5},force=s.force,build_check_type=defines.build_check_type.manual}
 local ok=game.surfaces[s.surface].can_place_entity(q)
 q.build_check_type=defines.build_check_type.ghost_revive
 bits[#bits+1]=(ok and game.surfaces[s.surface].can_place_entity(q)) and "1" or "0"
 end end out[#out+1]=table.concat(bits) end rcon.print(table.concat(out))"""
    )
    expected = (np.frombuffer(response.encode(), dtype=np.uint8) == ord("1")).reshape(
        63, 16, 16
    )
    np.testing.assert_array_equal(c.buildability.values, expected)


def test_build_destroy_and_tile_events_invalidate(world):
    rc, c = world
    fill(rc, c)
    channel = BuildabilityCache.channel_for("iron-chest", 0)
    assert c.buildability.values[channel, 8, 8] == 1
    rc.send_command(
        '/sc storage.cache_test_chest=game.surfaces[1].create_entity{name="iron-chest",position={2048.5,2048.5},force="player",raise_built=true} storage.obs_buildability.budget=128'
    )
    response = poll(rc, c)
    assert "R" in response
    assert c.buildability.values[channel, 8, 8] != 1
    rc.send_command("/sc storage.obs_buildability.budget=8192")
    fill(rc, c)
    assert c.buildability.values[channel, 8, 8] == 0
    rc.send_command("/sc storage.cache_test_chest.destroy{raise_destroy=true}")
    fill(rc, c)
    assert c.buildability.values[channel, 8, 8] == 1
    rc.send_command(
        '/sc game.surfaces[1].set_tiles({{name="water",position={2048,2048}}},true,false,false,true)'
    )
    # Explicit script tile event follows the normal event integration.
    rc.send_command(
        '/sc script.raise_event(defines.events.script_raised_set_tiles,{surface_index=1,tiles={{position={2048,2048},old_tile=prototypes.tile["grass-1"]}}})'
    )
    fill(rc, c)
    assert c.buildability.values[channel, 8, 8] == 0


def test_underground_fluid_changes_clear_and_recompute_direction_masks(world):
    rc, c = world
    fill(rc, c)
    rc.send_command("""/sc local s=game.surfaces[1]
storage.cache_water=s.create_entity{name="pipe",position={2048.5,2047.5},force="player",raise_built=true}
storage.cache_water.fluidbox[1]={name="water",amount=50}
storage.cache_steam=s.create_entity{name="pipe-to-ground",position={2048.5,2050.5},direction=8,force="player",raise_built=true}
storage.cache_steam.fluidbox[1]={name="steam",amount=50,temperature=100}
obs_diff_touch(storage.cache_steam)""")
    response = poll(rc, c)
    channel = BuildabilityCache.channel_for("pipe-to-ground", 0)
    assert f"J{c.buildability.generation},{channel}" in response
    fill(rc, c)
    assert [
        int(
            c.buildability.values[
                BuildabilityCache.channel_for("pipe-to-ground", d), 8, 8
            ]
        )
        for d in (0, 4, 8, 12)
    ] == [0, 1, 1, 1]
    rc.send_command(
        '/sc storage.cache_steam.fluidbox[1]={name="water",amount=50} obs_diff_touch(storage.cache_steam)'
    )
    fill(rc, c)
    assert c.buildability.values[channel, 8, 8] == 1


def test_full_sync_replays_cache_without_placement_queries(world):
    rc, c = world
    fill(rc, c)
    before = c.buildability.values.copy()
    generation = c.buildability.generation
    c.apply_terrain(rc.send_command("/sc obs_buildability_full_sync()"))
    assert c.buildability.generation > generation
    assert np.any(c.buildability.values == -1)
    for _ in range(5):
        assert (
            int(rc.send_command("/sc rcon.print(storage.obs_buildability.last_checks)"))
            == 0
        )
        poll(rc, c)
    np.testing.assert_array_equal(c.buildability.values, before)


def test_silent_mutation_is_found_by_rolling_refresh(world):
    rc, c = world
    fill(rc, c)
    rc.send_command(
        '/sc game.surfaces[1].create_entity{name="iron-chest",position={2048.5,2048.5},force="player"} storage.obs_buildability.refresh=1 game.tick_paused=false'
    )
    time.sleep(0.1)
    rc.send_command("/sc game.tick_paused=true")
    for _ in range(8):
        poll(rc, c)
    channel = BuildabilityCache.channel_for("iron-chest", 0)
    assert c.buildability.values[channel, 8, 8] == 0


def test_following_window_recenter_and_surface_change(world):
    rc, c = world
    rc.send_command(
        """/sc storage.obs_char=game.surfaces[1].create_entity{name="character",position={2048.5,2048.5},force="player"}
obs_buildability_configure{size=288,check_budget=128}"""
    )
    poll(rc, c)
    before, generation = c.buildability.origin, c.buildability.generation
    rc.send_command("/sc assert(storage.obs_char.teleport({2080.5,2048.5}))")
    poll(rc, c)
    assert c.buildability.origin == (before[0] + 32, before[1])
    assert c.buildability.origin == (c.center_x - 144, c.center_y - 144)
    assert c.buildability.generation == generation
    assert (
        int(rc.send_command("/sc rcon.print(storage.obs_buildability.last_checks)"))
        <= 128
    )
    # The client keeps independent state per surface; no old sampled bits leak.
    rc.send_command(
        """/sc local s=game.surfaces["cache-other"] or game.create_surface("cache-other",{})
s.request_to_generate_chunks({0,0},1) s.force_generate_chunk_requests()
assert(storage.obs_char.teleport({0,0},s))"""
    )
    poll(rc, c)
    assert c.buildability.generation > generation
    assert c.buildability.surface != 1
    assert (c.buildability.values >= 0).sum() <= 64
    rc.send_command(
        "/sc storage.obs_char.destroy{raise_destroy=true} storage.obs_char=nil"
    )


def test_dirty_work_does_not_starve_normal_traversal(world):
    rc, c = world
    fill(rc, c)
    rc.send_command("""/sc storage.obs_buildability.budget=128
local e=game.surfaces[1].create_entity{name="iron-chest",position={2048.5,2048.5},force="player",raise_built=true}
storage.cache_dirty_cursor=storage.obs_buildability.cursor""")
    for _ in range(6):
        poll(rc, c)
        assert (
            int(rc.send_command("/sc rcon.print(storage.obs_buildability.last_checks)"))
            <= 128
        )
    assert (
        rc.send_command(
            "/sc rcon.print(storage.obs_buildability.cursor~=storage.cache_dirty_cursor)"
        )
        == "true"
    )
