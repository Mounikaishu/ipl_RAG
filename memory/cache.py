
class CacheMemory:

    def __init__(self):

        self.memory = {}

    def get(
        self,
        query
    ):

        return self.memory.get(
            query
        )

    def set(
        self,
        query,
        answer
    ):

        self.memory[
            query
        ] = answer

        print(
            "\nAnswer Cached!"
        )