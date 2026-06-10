
import sys
from pathlib import Path
import re


# --------------------------
# CONNECT FRAMEWORK
# --------------------------

framework_path = (
    Path(__file__)
    .resolve()
    .parents[2]
    / "rag-pipeline-framework"
)

sys.path.insert(
    0,
    str(framework_path)
)

from generation.llm import (
    LLMModel
)


class IPLGenerator:

    def __init__(self):

        self.llm = (
            LLMModel()
            .get_llm()
        )

    # ------------------------
    # EXTRACT TABLE ROWS
    # ------------------------

    def extract_table_rows(
        self,
        context
    ):

        rows = []

        for line in context.split("\n"):

            if "|" not in line:
                continue

            if "---" in line:
                continue

            columns = [
                col.strip()
                for col in line.split("|")
                if col.strip()
            ]

            if len(columns) > 3:
                rows.append(columns)

        return rows

    # ------------------------
    # FIND MAX VALUE
    # ------------------------

    def get_max_stat(
        self,
        rows,
        stat_name
    ):

        try:

            headers = rows[0]

            stat_index = None

            for i, header in enumerate(headers):

                if (
                    stat_name.lower()
                    in header.lower()
                ):

                    stat_index = i
                    break

            if stat_index is None:
                return None

            best_player = None
            max_value = -1

            for row in rows[1:]:

                try:

                    player = row[0]

                    value = (
                        row[stat_index]
                        .replace("*", "")
                        .replace(",", "")
                    )

                    numeric_value = float(value)

                    if (
                        numeric_value
                        > max_value
                    ):

                        max_value = (
                            numeric_value
                        )

                        best_player = (
                            player
                        )

                except:
                    continue

            return (
                best_player,
                max_value
            )

        except:
            return None

    # ------------------------
    # MAIN ANSWER GENERATION
    # ------------------------

    def generate_answer(
        self,
        query,
        retrieved_docs,
        last_context=""
    ):

        context = (
            "\n\n".join(
                [
                    doc.page_content
                    for doc
                    in retrieved_docs

                    if hasattr(
                        doc,
                        "page_content"
                    )
                ]
            )
        )

        print(
            "\nRetrieved Context:\n"
        )

        print(
            context[:2000]
        )

        query_lower = (
            query.lower()
        )

        rows = (
            self.extract_table_rows(
                context
            )
        )

        # ------------------------
        # HIGHEST RUNS
        # ------------------------

        if (
            "run"
            in query_lower
            and any(
                word in query_lower
                for word in [
                    "highest",
                    "most",
                    "top",
                    "maximum"
                ]
            )
        ):

            result = (
                self.get_max_stat(
                    rows,
                    "Runs"
                )
            )

            if result:

                player, value = result

                return (
                    f"{player} "
                    f"has scored the "
                    f"highest number "
                    f"of IPL runs "
                    f"with {int(value)} runs."
                )

        # ------------------------
        # MOST WICKETS
        # ------------------------

        if (
            "wicket"
            in query_lower
            and any(
                word in query_lower
                for word in [
                    "highest",
                    "most",
                    "top",
                    "maximum"
                ]
            )
        ):

            result = (
                self.get_max_stat(
                    rows,
                    "Wickets"
                )
            )

            if result:

                player, value = result

                return (
                    f"{player} "
                    f"has taken the "
                    f"most IPL wickets "
                    f"with "
                    f"{int(value)} wickets."
                )

        # ------------------------
        # LLM FALLBACK
        # ------------------------

        prompt = f"""
You are an IPL RAG assistant.

Your job:
Answer ONLY using the
retrieved IPL context.

IMPORTANT:
Conversation history
matters.

If the user asks:

- they
- them
- those players
- those teams
- him
- her
- it

resolve the reference
using Previous Context.

STRICT RULES:

1. Use ONLY retrieved context

2. Use Previous Context
ONLY to resolve references

3. NEVER invent players

4. NEVER add outside
knowledge

5. NEVER mention players
not present in retrieved
context

6. Preserve filters exactly

Example:

Previous Context:
Players with more than
5000 runs:
Virat Kohli,
Rohit Sharma,
David Warner

Question:
How many centuries
they have scored?

Correct behavior:
Only answer for those
3 players.

If information is missing,
say:

"I could not find this
information in the IPL
documents."

Previous Context:
{last_context}

Question:
{query}

Retrieved Context:
{context}

Answer:
"""

        response = (
            self.llm.invoke(
                prompt
            )
        )

        return (
            response.content.strip()
        )

    # ------------------------
    # WEB ANSWER CLEANER
    # ------------------------

    def summarize_web_answer(
        self,
        query,
        web_result
    ):

        prompt = f"""
You are an assistant.

Answer ONLY using
the web result.

STRICT RULES:

1. Give short answer

2. Maximum 2 sentences

3. Extract ONLY relevant info

4. Ignore unrelated text

Question:
{query}

Web Result:
{web_result}

Clean Answer:
"""

        response = (
            self.llm.invoke(
                prompt
            )
        )

        return (
            response.content.strip()
        )
