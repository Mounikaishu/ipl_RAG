from generation.llm import (
    LLMModel
)

import re


class IPLGenerator:

    def __init__(self):

        self.llm = (
            LLMModel()
            .get_llm()
        )

    # ------------------------
    # TABLE PARSER
    # ------------------------

    def extract_rows(
        self,
        retrieved_docs
    ):

        rows = []

        batting_table_found = False

        for doc in retrieved_docs:

            if not hasattr(
                doc,
                "page_content"
            ):
                continue

            text = (
                doc.page_content
            )

            text_lower = (
                text.lower()
            )

            # ------------------------
            # ONLY USE BATTING TABLE
            # ------------------------

            if not (

                "player"
                in text_lower

                and "runs"
                in text_lower

                and "100s"
                in text_lower
            ):
                continue

            batting_table_found = True

            for line in text.split("\n"):

                line = (
                    line.strip()
                )

                if "|" not in line:
                    continue

                if "---" in line:
                    continue

                cols = [

                    col.strip()

                    for col in
                    line.split("|")

                    if col.strip()
                ]

                # batting table rows
                if len(cols) >= 9:

                    rows.append(cols)

        print(
            "\nParsed Rows:",
            len(rows)
        )

        if not batting_table_found:

            print(
                "\nCould not find "
                "batting table."
            )

        return rows

    # ------------------------
    # COLUMN FINDER
    # ------------------------

    def find_column(
        self,
        headers,
        target
    ):

        for i, col in enumerate(
            headers
        ):

            cleaned = (
                col.lower()
                .strip()
            )

            if (
                target.lower()
                in cleaned
            ):

                return i

        return None

    # ------------------------
    # MAIN ANSWER
    # ------------------------

    def generate_answer(
        self,
        query,
        retrieved_docs,
        last_context="",
        resolved_entities=None
    ):

        if (
            resolved_entities
            is None
        ):
            resolved_entities = []

        query_lower = (
            query.lower()
        )

        rows = (
            self.extract_rows(
                retrieved_docs
            )
        )

        print(
            "\nResolved Entities:",
            resolved_entities
        )

        if len(rows) < 2:

            return {
                "answer":
                (
                    "I could not find "
                    "this information "
                    "in the IPL documents."
                ),

                "resolved_entities":
                resolved_entities
            }

        # ------------------------
        # FIND HEADER
        # ------------------------

        headers = None

        for row in rows:

            joined = (
                " ".join(row)
                .lower()
            )

            if (
                "player"
                in joined
                and "runs"
                in joined
            ):

                headers = row
                break

        if not headers:

            return {

                "answer":
                "Could not find batting table.",

                "resolved_entities":
                resolved_entities
            }

        player_idx = (
            self.find_column(
                headers,
                "player"
            )
        )

        runs_idx = (
            self.find_column(
                headers,
                "runs"
            )
        )

        centuries_idx = (
            self.find_column(
                headers,
                "100"
            )
        )

        if (

            player_idx is None
            or runs_idx is None
        ):

            return {

                "answer":
                "Could not parse batting stats.",

                "resolved_entities":
                resolved_entities
            }

        # ------------------------
        # CLEAN DATA ROWS
        # ------------------------

        clean_rows = []

        already_seen = set()

        for row in rows:

            try:

                if (
                    len(row)
                    <= max(
                        player_idx,
                        runs_idx
                    )
                ):
                    continue

                player = (
                    row[
                        player_idx
                    ]
                    .strip()
                )

                # skip header row
                if (
                    player.lower()
                    == "player"
                ):
                    continue

                # remove duplicates
                if (
                    player
                    in already_seen
                ):
                    continue

                already_seen.add(
                    player
                )

                runs = int(
                    row[
                        runs_idx
                    ]
                    .replace(",", "")
                    .replace("*", "")
                )

                centuries = 0

                if (
                    centuries_idx
                    is not None
                    and len(row)
                    > centuries_idx
                ):

                    centuries = int(
                        row[
                            centuries_idx
                        ]
                    )

                clean_rows.append(
                    {
                        "player":
                        player,

                        "runs":
                        runs,

                        "centuries":
                        centuries
                    }
                )

            except:
                continue

        # ------------------------
        # RUNS QUERY
        # ------------------------

        if (
            "run"
            in query_lower
            and any(
                x in query_lower
                for x in [
                    "more than",
                    "greater than",
                    "above"
                ]
            )
        ):

            numbers = re.findall(
                r"\d+",
                query_lower
            )

            threshold = (
                int(numbers[0])
                if numbers
                else 0
            )

            filtered = []

            for row in clean_rows:

                if (
                    row["runs"]
                    > threshold
                ):

                    filtered.append(
                        row
                    )

            updated_entities = [

                row["player"]

                for row
                in filtered
            ]

            print(
                "\nUpdated Entities:",
                updated_entities
            )

            answer = "\n".join(

                f"{row['player']} - "
                f"{row['runs']} runs"

                for row
                in filtered
            )

            return {

                "answer":
                answer,

                "resolved_entities":
                updated_entities
            }

        # ------------------------
        # CENTURIES QUERY
        # ------------------------

        if (
            "centur"
            in query_lower
        ):

            filtered = []

            for row in clean_rows:

                # only previous players
                if (
                    len(
                        resolved_entities
                    ) > 0
                    and row[
                        "player"
                    ]
                    not in
                    resolved_entities
                ):
                    continue

                filtered.append(
                    row
                )

            answer = "\n".join(

                f"{row['player']} - "
                f"{row['centuries']} centuries"

                for row
                in filtered
            )

            return {

                "answer":
                answer,

                "resolved_entities":
                resolved_entities
            }

        # ------------------------
        # FALLBACK LLM
        # ------------------------

        context = "\n".join(

            doc.page_content

            for doc
            in retrieved_docs

            if hasattr(
                doc,
                "page_content"
            )

        )[:3000]

        prompt = f"""
Answer ONLY using IPL data.

Question:
{query}

Context:
{context}

Answer:
"""

        response = (
            self.llm.invoke(
                prompt
            )
        )

        return {

            "answer":
            response.content.strip(),

            "resolved_entities":
            resolved_entities
        }

    # ------------------------
    # WEB ANSWER
    # ------------------------

    def summarize_web_answer(
        self,
        query,
        web_result
    ):

        prompt = f"""
Question:
{query}

Web Result:
{web_result}

Short Answer:
"""

        response = (
            self.llm.invoke(
                prompt
            )
        )

        return (
            response.content
            .strip()
        )