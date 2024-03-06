# Building and Running Docker Image

## Building

```sh
$ docker build -t newhelm .
```

## Running

In order to run successfully, you will most likely want to mount a secrets directory with the required secrets and an output directory to store the output. You may also way to mount the run data directory which will help with caching in-between runs.

```sh
$ docker run \
    --mount type=bind,source=$(pwd)/run_data,target=/app/run_data \
    --mount type=bind,source=$(pwd)/secrets,target=/app/secrets \
    --mount type=bind,source=$(pwd)/output,target=/app/output \
      newhelm run-test \
        --test real_toxicity_prompts \
        --sut llama-2-7b \
        --max-test-items 1
```
