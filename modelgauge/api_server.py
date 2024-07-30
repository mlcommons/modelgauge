import itertools
import multiprocessing
import multiprocessing.pool
import os
from typing import Sequence, Optional

from fastapi import FastAPI, Depends, HTTPException  # type: ignore
from fastapi.security import APIKeyHeader  # type: ignore
from pydantic import BaseModel

from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.config import load_secrets_from_config
from modelgauge.load_plugins import load_plugins
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import PromptResponseSUT
from modelgauge.sut_registry import SUTS
from modelgauge.suts.together_client import CHAT_MODELS

"""
  Simple API server for modelgauge functionality. Currently used just for interviews.
  
  Start it up with something like `fastapi run modelgauge/api_server.py`
  
  To use it, GET / will show the list of available SUTs. Then you can POST / with
  something like:
  
  ```
  {
    "prompts": [{"text": "What's your name?","options": {"max_tokens": 50}}],
    "suts":["llama-2-7b-chat"]
  }
  ```
Multiple SUTs are allowed, and are run in parallel.
"""

load_plugins()

secrets = load_secrets_from_config()

suts: dict[str, PromptResponseSUT] = {
    sut_uid: SUTS.make_instance(sut_uid, secrets=secrets)  # type:ignore
    for sut_uid in CHAT_MODELS.keys()
}

annotators: dict[str, CompletionAnnotator] = {
    sut_uid: ANNOTATORS.make_instance(sut_uid, secrets=secrets)  # type:ignore
    for sut_uid in [i[0] for i in ANNOTATORS.items()]
}

print(f"got suts {suts} and annotators {annotators}")


class ProcessingRequest(BaseModel):
    prompts: Sequence[TextPrompt]
    suts: Sequence[str]
    annotators: Sequence[str] = []


SECRET_KEY = os.getenv("SECRET_KEY")
assert SECRET_KEY, "must set SECRET_KEY environment variable"
app = FastAPI()


@app.get("/")
async def get_options():
    return {"suts": list(suts.keys()), "annotators": list(annotators.keys())}


def process_work_item(
    prompt: TextPrompt, sut_key: str, annotator_key: Optional[str] = None
):
    sut = suts[sut_key]
    s_req = sut.translate_text_prompt(prompt)
    s_resp = sut.translate_response(s_req, sut.evaluate(s_req))
    result = {"sut": sut.uid, "sut_response": s_resp}
    if annotator_key:
        annotator = annotators[annotator_key]
        a_req = annotator.translate_request(
            PromptWithContext(prompt=prompt, source_id="whatever, man"),
            s_resp.completions[0],
        )
        result["annotation"] = annotator.translate_response(
            a_req, annotator.annotate(a_req)
        )
    return result


auth_header = APIKeyHeader(name="x-key")


@app.post("/")
async def process_sut_request(req: ProcessingRequest, key: str = Depends(auth_header)):
    if key != SECRET_KEY:
        raise HTTPException(401, "not authorized; send x-key header")
    for sut in req.suts:
        if not sut in suts:
            raise HTTPException(422, f"sut {sut} not found")
    if req.annotators:
        work_items = list(itertools.product(req.prompts, req.suts, req.annotators))
    else:
        work_items = list(itertools.product(req.prompts, req.suts))  # type:ignore

    print(work_items)
    results = await process_work_items(work_items)
    return {"request": req, "response": results}


async def process_work_items(work_items):
    worker_count = len(work_items) or 1
    pool = multiprocessing.pool.ThreadPool(worker_count)
    results = pool.starmap(process_work_item, work_items)
    return results
