# Pinned live vocabulary

Generate the base-game artifact from a running Factorio 2.0.73 `open_world`
server after validating its mod list:

```sh
python -m fle.rl.export_vocabulary --host 127.0.0.1 --port 27000
```

This writes `factorio-2.0.73-base-v1.json` only for a base/core-only mod set and
refuses to overwrite an existing artifact. Check every training server before
reset with:

```sh
python -m fle.rl.export_vocabulary --host 127.0.0.1 --port 27000 --check data/rl/vocab/factorio-2.0.73-base-v1.json
```

The JSON fixture under `tests/rl/fixtures/` is hand-written test data, not a
live game catalog. The four V0 goal items remain candidates until automatic
counter and manual-ledger calibration on a live server.
