#!/usr/bin/env bash

export TQDM_DISABLE=1

CMD="$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )/../newhelm/main.py"

declare -a TESTS=("demo_01" "demo_02")
declare -a SUTS=("DemoMultipleChoiceSUT")

echo "Test main"
python $CMD > /dev/null

echo "Test list"
python $CMD list > /dev/null

echo "Test run-test test sut combinations"
for test in "${TESTS[@]}"; do
  for sut in "${SUTS[@]}"; do
    echo "Test $test with $sut"
    python $CMD run-test \
      --test $test \
      --sut $sut \
      --max-test-items 1 > /dev/null
  done
done
