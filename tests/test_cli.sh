#!/usr/bin/env bats

setup() {
  CMD="$( cd "$( dirname "$BATS_TEST_FILENAME" )" >/dev/null 2>&1 && pwd )/../newhelm/main.py"
  declare -a TESTS=("demo_01" "demo_02")
  declare -a SUTS=("DemoMultipleChoiceSUT")
}

@test "main" {
  run python $CMD
  [ "$status" -eq 0 ]
}

@test "list" {
  run python $CMD list
  [ "$status" -eq 0 ]
}

@test "test sut combinations" {
  for test in "${TESTS[@]}"; do
    for sut in "${SUTS[@]}"; do
      run python $CMD run-test \
        --test $test \
        --sut $sut \
        --max-test-items 1
      [ "$status" -eq 0 ]
    done
  done
}
