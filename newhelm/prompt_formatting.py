from newhelm.prompt import ChatPrompt, ChatRole


def format_chat(
    chat: ChatPrompt, *, user_text: str = "user: ", sut_text: str = "assistant: "
) -> str:
    """Flattens as chat conversation into a single text prompt"""
    blocks = []
    for message in chat.messages:
        role_text: str
        if message.role == ChatRole.user:
            role_text = user_text
        else:
            role_text = sut_text
        blocks.append(f"{role_text}{message.text}")
    blocks.append(sut_text)
    return "\n\n".join(blocks)
