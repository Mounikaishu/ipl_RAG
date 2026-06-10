
from collections import (
    deque
)


class ConversationMemory:

    def __init__(
        self,
        max_history=5
    ):

        self.history = deque(
            maxlen=max_history
        )

    def add_message(
        self,
        role,
        content
    ):

        self.history.append(
            {
                "role":
                role,

                "content":
                content
            }
        )

    def get_history(
        self
    ):

        return list(
            self.history
        )

    def get_context(
        self
    ):

        if (
            not self.history
        ):

            return ""

        conversation = []

        for msg in (
            self.history
        ):

            conversation.append(
                f"{msg['role']}: "
                f"{msg['content']}"
            )

        return "\n".join(
            conversation
        )
