# Safety Models Plugin
**This plugin is in alpha mode. Expect there to be issues. Please reach out to the engineering team to resolve**

## Notes
- This test is currently only compatible with the 1320 MLC human annotated dataset, which is under restricted access (contact engineering team for access)
- This test requires using the annotator specific test runner, which is not configurable except by code.

## Known issues
- running pytests using zsh (instead of bash) as your terminal has issues collecting tests due to the * wildcard search issue. TLDR: use bash instead to run pytests
