import logging
import random
import threading
import time
from typing import Dict, List, Mapping, Optional, Sequence
from googleapiclient import discovery  # type: ignore[import-untyped]
from googleapiclient.http import BatchHttpRequest  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
from pydantic import BaseModel  # type: ignore[import-untyped]

from newhelm.base_annotator import BaseAnnotator
from newhelm.secret_values import RequiredSecret, SecretDescription
from newhelm.single_turn_prompt_response import PromptInteraction, PromptWithContext
from newhelm.sut import SUTCompletion

# Constants for requesting specific kinds of scoring from Perspective API.
# https://developers.perspectiveapi.com/s/about-the-api-attributes-and-languages
ATTRIBUTE_TOXICITY = "TOXICITY"
ATTRIBUTE_SEVERE_TOXICITY = "SEVERE_TOXICITY"
ATTRIBUTE_IDENTITY_ATTACK = "IDENTITY_ATTACK"
ATTRIBUTE_INSULT = "INSULT"
ATTRIBUTE_PROFANITY = "PROFANITY"
ATTRIBUTE_THREAT = "THREAT"
ATTRIBUTE_SEXUALLY_EXPLICIT = "SEXUALLY_EXPLICIT"
ATTRIBUTE_FLIRTATION = "FLIRTATION"


class PerspectiveAPIAnnotation(BaseModel):
    class Completion(BaseModel):
        scores: Mapping[str, float]
        """Mapping from ATTRIBUTE to score."""

    class Interaction(BaseModel):
        completions: Sequence["PerspectiveAPIAnnotation.Completion"]
        """Each element corresponds to a completion to a single Prompt."""

    interaction: Sequence[Interaction]
    """Each element in the list corresponds to a Prompt in the TestItem."""


class PerspectiveDeveloperKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="perspective_api",
            key="api_key",
            instructions="First request access https://developers.perspectiveapi.com/s/docs-get-started?language=en_US"
            " and then you can generate a key with https://developers.perspectiveapi.com/s/docs-enable-the-api?language=en_US",
        )


class AnalyzeCommentRequest(BaseModel):
    # https://developers.perspectiveapi.com/s/about-the-api-methods
    class Comment(BaseModel):
        text: str
        type: Optional[str] = None

    comment: Comment

    class RequestedAttribute(BaseModel):
        scoreType: Optional[str] = None
        scoreThreshold: Optional[float] = None

    requestedAttributes: Mapping[str, RequestedAttribute]
    languages: Optional[Sequence[str]]

    class Context(BaseModel):
        entries: Optional[Sequence["AnalyzeCommentRequest.ContextEntry"]] = None

    class ContextEntry(BaseModel):
        text: Optional[str] = None
        type: Optional[str] = None

    context: Optional[Context] = None
    spanAnnotations: Optional[bool] = None
    doNotStore: Optional[bool] = None
    clientToken: Optional[str] = None
    sessionId: Optional[str] = None
    communityId: Optional[str] = None


class AnalyzeCommentResponse(BaseModel):
    # https://developers.perspectiveapi.com/s/about-the-api-methods

    class AttributeScore(BaseModel):
        summaryScore: "AnalyzeCommentResponse.Score"
        spanScores: Optional[Sequence["AnalyzeCommentResponse.SpanScore"]] = None

    class Score(BaseModel):
        value: float
        type: Optional[str] = None

    class SpanScore(BaseModel):
        begin: int
        end: int
        score: "AnalyzeCommentResponse.Score"

    attributeScores: Mapping[str, AttributeScore]
    languages: Optional[Sequence[str]] = None
    clientToken: Optional[str] = None


class PerspectiveAPIAnnotator(BaseAnnotator[PerspectiveAPIAnnotation]):
    """Collect information about a SUT's response using PerspectiveAPI 'analyze comment' interface."""

    def __init__(
        self,
        desired_attributes: List[str],
        developer_key: PerspectiveDeveloperKey,
        languages: Optional[List[str]] = None,
        num_retries: int = 6,
    ):
        assert desired_attributes, "Must specify at least one desired attribute"
        assert len(desired_attributes) == len(
            set(desired_attributes)
        ), "All desired attributes should be unique."
        self.attributes = desired_attributes
        self.developer_key = developer_key.value
        self.languages = languages
        self.num_retries = num_retries
        self.rng = random.Random()  # Used for exponential backoff
        self.client: Optional[discovery.Resource] = None
        # httplib2 is not thread-safe. Acquire this lock when sending requests to PerspectiveAPI
        self._client_lock: threading.Lock = threading.Lock()

    def _load_client(self) -> discovery.Resource:
        return discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.developer_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def annotate_test_item(
        self, interactions: List[PromptInteraction]
    ) -> PerspectiveAPIAnnotation:
        with self._client_lock:
            # Handle lazy init.
            if self.client is None:
                self.client = self._load_client()

        analyze_requests: List[AnalyzeCommentRequest] = []
        empty_completion_indexes = set()
        request_index = 0
        for interaction in interactions:
            for completion in interaction.response.completions:
                if completion.text == "":
                    empty_completion_indexes.add(request_index)
                else:
                    analyze_requests.append(
                        self._make_analyze_comment_request(completion.text)
                    )
                request_index += 1

        with self._client_lock:
            requests = []
            for analyze_request in analyze_requests:
                body = analyze_request.model_dump(exclude_none=True)
                requests.append(self.client.comments().analyze(body=body))
            responses = _batch_execute_requests(
                self.client, requests, self.num_retries, self.rng
            )
            analyze_responses: List[AnalyzeCommentResponse] = []
            for response in responses:
                analyze_responses.append(
                    AnalyzeCommentResponse.model_validate(response)
                )

        index = 0
        response_index = 0
        interaction_scores = []
        for interaction in interactions:
            completion_scores = []
            for _ in interaction.response.completions:
                if index in empty_completion_indexes:
                    completion_scores.append(
                        PerspectiveAPIAnnotation.Completion(
                            scores={attribute: 0.0 for attribute in self.attributes}
                        )
                    )
                else:
                    completion_scores.append(
                        PerspectiveAPIAnnotation.Completion(
                            scores=self._process_analyze_comment_response(
                                analyze_responses[response_index]
                            )
                        )
                    )
                    response_index += 1
                index += 1
            interaction_scores.append(
                PerspectiveAPIAnnotation.Interaction(completions=completion_scores)
            )
        return PerspectiveAPIAnnotation(interaction=interaction_scores)

    def _make_analyze_comment_request(self, completion: str):
        requested_attributes = {
            attribute: AnalyzeCommentRequest.RequestedAttribute()
            for attribute in self.attributes
        }
        return AnalyzeCommentRequest(
            # TODO: Consider what to do if text is too long.
            comment=AnalyzeCommentRequest.Comment(text=completion),
            # TODO: Consider setting the "context" field based on the prompt's text.
            requestedAttributes=requested_attributes,
            languages=self.languages,
            # HELM sets spanAnnotations=True
        )

    def _process_analyze_comment_response(
        self, response: AnalyzeCommentResponse
    ) -> Dict[str, float]:
        flattened: Dict[str, float] = {
            # This differs from HELM by using summaryScore.
            attribute: response.attributeScores[attribute].summaryScore.value
            for attribute in self.attributes
        }
        return flattened


def _batch_execute_requests(
    client: discovery.Resource, requests: List, num_retries: int, rng: random.Random
) -> List:
    """Wrapper around Google's batch API.

    This can give significant speedup. For example for PerspectiveAPI, batching
    25 requests is about 15x faster than doing each as separate calls.
    https://googleapis.github.io/google-api-python-client/docs/batch.html
    """

    if not requests:
        return []

    errors = [None] * len(requests)
    responses: List[Dict] = [{}] * len(requests)

    def _callback(request_id: str, response: Dict, error: HttpError):
        index = int(request_id)
        if error:
            errors[index] = error
        else:
            # Clear any past errors
            errors[index] = None
        responses[index] = response

    # Keep track of what requests have not yet successfully gotten a response
    needs_call = list(range(len(requests)))
    retriable_errors: List[HttpError] = []
    for retry_count in range(num_retries + 1):
        if retry_count > 0:
            # Perform exponential backoff
            sleep_amount = rng.uniform(1, 2) * 2**retry_count
            logging.info("Performing exponential backoff. Sleeping:", sleep_amount)
            time.sleep(sleep_amount)

        # Build up a batch
        batch_request: BatchHttpRequest = client.new_batch_http_request()
        for i in needs_call:
            batch_request.add(
                request=requests[i],
                request_id=str(i),
                callback=_callback,
            )
        # Fills up responses using the callback
        batch_request.execute()

        # Figure out which requests need to be tried again.
        next_round_needs_call: List[int] = []
        fatal_errors: List[HttpError] = []
        retriable_errors = []
        for i in needs_call:
            error = errors[i]
            if error is not None:
                if _is_retriable(error):
                    next_round_needs_call.append(i)
                    retriable_errors.append(error)
                else:
                    fatal_errors.append(error)
        if fatal_errors:
            # Just use the first one as an example.
            raise fatal_errors[0]
        if not next_round_needs_call:
            break
        needs_call = next_round_needs_call
    if retriable_errors:
        # We exhausted our retries, so raise the first as an example.
        raise retriable_errors[0]
    return responses


def _is_retriable(error: HttpError) -> bool:
    """Check if this error can be retried."""
    # Retry any 5XX status.
    if 500 <= error.status_code < 600:
        return True
    # 429 is "Too Many Requests" and for PerspectiveAPI means "RATE_LIMIT_EXCEEDED"
    if error.status_code == 429:
        return True
    return False


if __name__ == "__main__":
    import sys
    from newhelm.prompt import TextPrompt
    from newhelm.single_turn_prompt_response import PromptWithContext
    from newhelm.config import load_secrets_from_config
    from newhelm.sut import SUTResponse

    sut_text = sys.argv[1]
    print("Sending:", sut_text)
    secrets = load_secrets_from_config()
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY, ATTRIBUTE_THREAT], PerspectiveDeveloperKey.make(secrets)
    )
    interactions: List[PromptInteraction] = [
        PromptInteraction(
            prompt=PromptWithContext(
                prompt=TextPrompt(text="The prompt"), source_id=None
            ),
            response=SUTResponse(completions=[SUTCompletion(text=sut_text)]),
        )
    ]
    response = annotator.annotate_test_item(interactions)
    print(response)
