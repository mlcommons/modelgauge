from modelgauge.prompt import ChatPrompt, ChatRole


def flatten_conversation_to_str(
    chat: ChatPrompt, *, user_role: str = "User", sut_role: str = "Assistant"
) -> str:
    """Flattens a chat conversation into a single text prompt"""
    blocks = []
    for message in chat.messages:
        role_text: str
        if message.role == ChatRole.user:
            role_text = user_role
        else:
            role_text = sut_role
        blocks.append(f"{role_text}: {message.text}")
    return "\n\n".join(blocks)
