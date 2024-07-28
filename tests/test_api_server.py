import os
from unittest.mock import patch

from starlette.testclient import TestClient  # type: ignore


class TestApiApp:
    def setup_method(self):
        real_getenv = os.getenv
        with patch(
            "os.getenv",
            lambda *args: "whatever" if args[0] == "SECRET_KEY" else real_getenv(*args),
        ):
            with patch(
                "modelgauge.config.load_secrets_from_config",
                lambda: {"together": {"api_key": "ignored"}},
            ):
                import modelgauge.api_server

                self.client = TestClient(modelgauge.api_server.app)

    def test_read_main(self):
        response = self.client.get("/")
        assert response.status_code == 200

        j = response.json()
        assert "llama_guard_1" in j["annotators"]
        assert "llama-2-7b-chat" in j["suts"]
