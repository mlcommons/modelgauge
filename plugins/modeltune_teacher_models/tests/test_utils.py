from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole
from modelgauge.safety_model_utils import flatten_conversation_to_str


def test_flatten_conversation_to_str():
    chat = ChatPrompt(
        messages=[
            ChatMessage(role=ChatRole.user, text="Hello"),
            ChatMessage(role=ChatRole.sut, text="Hi!"),
        ]
    )

    result = flatten_conversation_to_str(chat)

    assert result == "User: Hello\n\nAssistant: Hi!"
