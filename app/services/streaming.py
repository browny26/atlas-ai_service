"""Incremental extraction of completed days from a JSON answer that is still being streamed.

The model writes {"days": [{...day 1...}, {...day 2...}, ...], ...} token by token. As soon
as a day object is closed we can validate it and send it to the client, long before the
whole answer is finished.
"""


class DayStreamParser:
    """Finds the fully-written objects of the top-level "days" array, chunk by chunk.

    A small state machine instead of a regex: it tracks strings (and escapes inside them)
    and nesting depth, so braces inside activity names cannot confuse it. The state is
    kept between calls, so each character is scanned once: the total work is linear in
    the answer length, however many chunks it arrives in.
    """

    def __init__(self) -> None:
        self._text: list[str] = []  # characters of the day currently being written
        self._depth = 0
        self._in_string = False
        self._escaped = False
        self._string: list[str] = []  # current depth-1 string (a possible key)
        self._last_key = ""
        self._expecting_key = False
        self._in_days = False
        self._in_day = False

    def feed(self, chunk: str) -> list[str]:
        """Consume the next chunk; return the raw JSON of the days completed in it."""
        completed: list[str] = []
        for char in chunk:
            if self._in_day:
                self._text.append(char)

            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_string = False
                    if self._depth == 1 and self._expecting_key:
                        self._last_key = "".join(self._string)
                elif self._depth == 1:
                    self._string.append(char)
                continue

            if char == '"':
                self._in_string = True
                self._string = []
            elif char in "{[":
                self._depth += 1
                if self._depth == 1:
                    self._expecting_key = True
                elif self._depth == 2 and char == "[" and self._last_key == "days":
                    self._in_days = True
                elif self._depth == 3 and char == "{" and self._in_days:
                    self._in_day = True
                    self._text = [char]
            elif char in "}]":
                if self._depth == 3 and char == "}" and self._in_day:
                    completed.append("".join(self._text))
                    self._in_day = False
                elif self._depth == 2 and char == "]":
                    self._in_days = False
                self._depth -= 1
            elif self._depth == 1:
                if char == ":":
                    self._expecting_key = False
                elif char == ",":
                    self._expecting_key = True
        return completed


def extract_completed_days(text: str) -> list[str]:
    """Every fully-written day object in `text` (one-shot use of the parser)."""
    return DayStreamParser().feed(text)
