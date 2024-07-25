import click
from typing import List

from modelgauge.annotation_pipeline import (
    AnnotatorAssigner,
    AnnotatorSink,
    AnnotatorSource,
    AnnotatorWorkers,
    CsvAnnotatorInput,
    JsonlAnnotatorOutput,
)
from modelgauge.pipeline import Pipeline, PipelineSegment
from modelgauge.prompt_pipeline import (
    PromptSource,
    PromptSutAssigner,
    PromptSutWorkers,
    PromptSink,
    CsvPromptInput,
    CsvPromptOutput,
)
from modelgauge.sut_capabilities import AcceptsTextPrompt


def build_prompt_pipeline_segments(
    suts,
    input_path,
    output_path=None,
    workers=None,
    sut_cache_dir=None,
    include_sink=True,
):
    """Returns an in-order list of pipeline segments required for a prompt pipeline."""
    segments: List[PipelineSegment] = []
    input = CsvPromptInput(input_path)
    segments.append(PromptSource(input))
    segments.append(PromptSutAssigner(suts))
    segments.append(PromptSutWorkers(suts, workers, cache_path=sut_cache_dir))
    if include_sink:
        assert (
            output_path is not None
        ), "output_path must be provided if include_sink=True."
        output = CsvPromptOutput(output_path, suts)
        segments.append(PromptSink(suts, output))
    return segments


def build_annotator_pipeline_segments(
    annotators, output_path, input_path=None, workers=None, include_source=True
):
    """Returns an in-order list of pipeline segments required for an annotator pipeline."""
    segments: List[PipelineSegment] = []
    if include_source:
        assert (
            input_path is not None
        ), "input_path must be provided if include_source=True."
        input = CsvAnnotatorInput(input_path)
        segments.append(AnnotatorSource(input))
    segments.append(AnnotatorAssigner(annotators))
    segments.append(AnnotatorWorkers(annotators, workers))
    output = JsonlAnnotatorOutput(output_path)
    segments.append(AnnotatorSink(annotators, output))
    return segments


def run_prompts_pipeline(
    suts, workers, sut_cache_dir, debug, input_path, output_path, annotators=None
):
    """Runs a pipeline for a given CSV file of prompts over a set of SUTs."""
    run_annotators = annotators is not None and len(annotators)

    if sut_cache_dir:
        print(f"Creating cache dir {sut_cache_dir}")
        sut_cache_dir.mkdir(exist_ok=True)

    for sut_uid in suts:
        sut = suts[sut_uid]
        if not AcceptsTextPrompt in sut.capabilities:
            raise click.BadParameter(f"{sut_uid} does not accept text prompts")

    if run_annotators:
        # Adds annotator pipes after prompt pipes. Uses annotator output sink instead of regular prompt output.
        pipeline_segments = build_prompt_pipeline_segments(
            suts,
            input_path,
            workers=workers,
            sut_cache_dir=sut_cache_dir,
            include_sink=False,
        )

        pipeline_segments.extend(
            build_annotator_pipeline_segments(
                annotators, output_path, workers=workers, include_source=False
            )
        )
        num_input_items = len(pipeline_segments[0].input)
        total_items = num_input_items * len(suts) * len(annotators)
        print(
            f"Processing {num_input_items} prompts * {len(suts)} SUTS * {len(annotators)} annotators."
        )
    else:
        pipeline_segments = build_prompt_pipeline_segments(
            suts,
            input_path,
            output_path,
            workers=workers,
            sut_cache_dir=sut_cache_dir,
            include_sink=True,
        )
        num_input_items = len(pipeline_segments[0].input)
        total_items = num_input_items * len(suts)
        print(f"Processing {num_input_items} prompts * {len(suts)} SUTS)")

    run_pipeline(pipeline_segments, total_items, debug)
    print(f"output saved to {output_path}")


def run_annotator_pipeline(annotators, workers, debug, input_path, output_path):
    pipeline_segments = build_annotator_pipeline_segments(
        annotators, output_path, input_path=input_path, workers=workers
    )
    num_input_items = len(pipeline_segments[0].input)
    total_items = num_input_items * len(annotators)
    print(f"Processing {num_input_items} items * {len(annotators)} annotators.")

    run_pipeline(pipeline_segments, total_items, debug)
    print(f"output saved to {output_path}")


def run_pipeline(pipeline_segments, total_items, debug):
    with click.progressbar(length=total_items) as bar:
        last_complete_count = 0

        def show_progress(data):
            nonlocal last_complete_count
            complete_count = data["completed"]
            bar.update(complete_count - last_complete_count)
            last_complete_count = complete_count

        pipeline = Pipeline(
            *pipeline_segments,
            progress_callback=show_progress,
            debug=debug,
        )

        pipeline.run()
