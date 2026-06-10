class Validator:

    def validate(
        self,
        query,
        answer
    ):

        print(
            "\nValidating answer..."
        )

        if (
            not answer
            or len(answer.strip())
            == 0
        ):

            return (
                "I could not find "
                "this information."
            )

        return answer