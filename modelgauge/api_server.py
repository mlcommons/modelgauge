import itertools
import multiprocessing
import time
from typing import Sequence

from fastapi import FastAPI
from pydantic import BaseModel

from modelgauge.config import load_secrets_from_config
from modelgauge.load_plugins import load_plugins
from modelgauge.sut import PromptResponseSUT
from modelgauge.sut_registry import SUTS
from modelgauge.suts.together_client import CHAT_MODELS
from prompt import TextPrompt

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
    sut_uid: SUTS.make_instance(sut_uid, secrets=secrets)
    for sut_uid in CHAT_MODELS.keys()
}

print(f"got suts {suts}")


class ProcessingRequest(BaseModel):
    prompts: Sequence[TextPrompt]
    suts: Sequence[str]


app = FastAPI()


@app.get("/")
async def root():
    return {"suts": list(suts.keys())}


def process_work_item(prompt: TextPrompt, sut: PromptResponseSUT):
    start = time.time()
    request = sut.translate_text_prompt(prompt)
    response = sut.evaluate(request)
    print(f"{sut.uid} took {time.time() - start}")
    return {"sut": sut.uid, "response": sut.translate_response(request, response)}


@app.post("/")
async def postroot(req: ProcessingRequest):
    work_items = list(itertools.product(req.prompts, [suts[k] for k in req.suts]))
    print(work_items)
    pool = multiprocessing.pool.ThreadPool(len(work_items))
    results = pool.starmap(process_work_item, work_items)
    return {"request": req, "result": results}
