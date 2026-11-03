"""Incremental extraction of completed days from a JSON answer that is still being streamed.

The model writes {"days": [{...day 1...}, {...day 2...}, ...], ...} token by token. As soon
as a day object is closed we can validate it and send it to the client, long before the
whole answer is finished.
"""


def extract_completed_days(text: str) -> list[str]:
    """Return the raw JSON of every fully-written object in the top-level "days" array.

    A small state machine instead of a regex: it tracks strings (and escapes inside them)
    and nesting depth, so braces inside activity names cannot confuse it.
    """
    days: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    string_start = -1
    last_key = ""
    expecting_key = False
    in_days = False
    day_start = -1

    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                if depth == 1 and expecting_key:
                    last_key = text[string_start + 1 : i]
            continue

        if char == '"':
            in_string = True
            string_start = i
        elif char in "{[":
            depth += 1
            if depth == 1:
                expecting_key = True
            elif depth == 2 and char == "[" and last_key == "days":
                in_days = True
            elif depth == 3 and char == "{" and in_days:
                day_start = i
        elif char in "}]":
            if depth == 3 and char == "}" and in_days and day_start >= 0:
                days.append(text[day_start : i + 1])
                day_start = -1
            elif depth == 2 and char == "]":
                in_days = False
            depth -= 1
        elif depth == 1:
            if char == ":":
                expecting_key = False
            elif char == ",":
                expecting_key = True
    return days
