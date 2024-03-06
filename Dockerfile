from python:3.10-slim-bookworm

RUN pip install poetry==1.7.1
WORKDIR /app
COPY . /app
RUN poetry install --extras all_plugins
ENTRYPOINT ["poetry", "run", "python", "newhelm/main.py"]
