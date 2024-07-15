# Safety Models Plugin
**This plugin is in alpha mode. Expect there to be issues. Please reach out to the engineering team to resolve**

## Notes
- This test is currently only compatible with the 1320 MLC human annotated dataset, which is under restricted access (contact engineering team for access)
- This test requires using the annotator specific test runner, which is not configurable except by code.

## Known issues
- running pytests using zsh (instead of bash) as your terminal has issues collecting tests due to the * wildcard search issue. TLDR: use bash instead to run pytests

## Steps to run
1. get access to and download 1320 dataset
1. add the 1320 dataset to modelgauge repo root dir, rename it to `1320mlc.csv`
1. add together api secret (if using together api annotators)
    1. open config/secrets.toml, under [together] api-key add your together api key
1. set up dependencies
1. `poetry install --extras all_plugins`
1. run `poetry run modelgauge run-annotator-test --test safety_eval_1320 --annotator llama_guard_2 --no-caching`