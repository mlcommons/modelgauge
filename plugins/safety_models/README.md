# Safety Models Plugin
**This plugin is in alpha mode. Expect there to be issues. Please reach out to the engineering team to resolve**


## Notes
- This test is currently only compatible with the 1320 MLC human annotated dataset, which is under restricted access (contact engineering team for access)

## Getting started
_Steps confirmed as of: 6/24/24_
1. **Get access to the 1320 dataset**
    - This is currently the only valid input dataset
1. **Upload 1320 dataset locally**
    - Add the dataset to the root directory of modelgauge
    - Name it `1320mlc.csv`
1. **Register the test**
    - In `plugins/safety_models/modelgauge/tests/eval_model_test.py`, add the following line at the bottom to register the test with modelgauge

    ```
    TESTS.register(EvalModelTest, "lg_eval_1320", "1320mlc.csv")

    ```
1. **Add together API key**
    - This run uses the together Llama Guard 2 endpoint for inference, so you'll need a together API key to make this work
    - (Follow the instructions here (TODO add link) to add your API key... just make sure the secret is added and discoverable)
1. **Verify the new test is registered**
    - Use the following command to validate if the new test was registered with modelgauge.
    ```
    poetry run modelgauge list-tests

    ```
    - verify that `lg_eval_1320` is one of the listed tests
1. **Run the test**
    ```
    poetry run modelgauge run-test --test lg_eval --sut YOUR_SUT
    ```
    - TODO add notes on which SUTs are supported since we'll only support safety models


## Known issues
- running pytests using zsh (instead of bash) as your terminal has issues collecting tests due to the * wildcard search issue. TLDR: use bash instead to run pytests
