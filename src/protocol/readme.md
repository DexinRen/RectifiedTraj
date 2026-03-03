# Protocol Index

This folder defines the engineering contract for the core trajectory stack.
Treat these files as the authoritative specification before editing code.

## Canonical protocol files
- `theta_model.h`: architecture contract for `src/theta_model.py`
- `theta_train.h`: training/runtime contract for `src/theta_train.py`
- `encoder_decoder.h`: inference contract for `src/encoder_decoder.py`
- `baseline.txt`: baseline pipeline contract
- `data_processing.txt`: data pipeline contract

## Current storage layout (active)
- Model root is hypothesis-aware:
  - `./bin/model/RectifiedTraj/<model_name>/...`
  - `./bin/model/ResidualReg/<model_name>/...`
- Training artifacts per model:
  - `ckpts/`
  - `log/config.json`
  - `log/config_init.json`
  - `log/train_data.csv`
  - `fig/*.png`
- Global runtime log:
  - `./bin/log/theta_train.log`

## Runtime and style law
- Use `runtime` dict for shared training state.
- Keep runtime namespace clean: only globally relevant keys.
- Use readable section separators and function headers.
- Use hybrid OOP + functional design:
  - Class for stateful loaders/components.
  - Functions for orchestration and stateless helpers.
- Use hard-fail behavior:
  - Do not swallow errors with defensive try/catch wrappers.
  - Validate at boundaries; avoid redundant deep re-validation.

## Change policy
- If implementation diverges from protocol, update protocol first or in the same change.
- If a planned feature changes the contract (for example hypothesis-specific training semantics), discuss and approve before code lands.
