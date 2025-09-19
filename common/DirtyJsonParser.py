import re
import dirtyjson


class DirtyJsonParser:
    """
    Attempts to extract the first JSON object from a raw VLM response,
    stripping out markdown fences (```json```) or any leading/trailing text.
    Falls back to dirty_json.loads for tolerant parsing.
    """

    @staticmethod
    def _extract_braced_block(text: str) -> str:
        """
        Find the first balanced JSON-like {...} block in `text`, using a simple stack
        approach. Returns the substring including the outermost braces.
        """
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("No opening brace '{' found in text.")

        stack = []
        for idx in range(start_idx, len(text)):
            char = text[idx]
            if char == '{':
                stack.append('{')
            elif char == '}':
                if not stack:
                    # Unbalanced closing brace; skip it
                    continue
                stack.pop()
                if not stack:
                    # All braces closed; return substring
                    return text[start_idx : idx + 1]

        raise ValueError("No matching closing '}' found for first '{' in text.")

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """
        Remove Markdown-style fences (```json ... ```) if present, along with any prefix/suffix.
        """
        # Remove any ```json and ``` fences (single or triple backticks)
        # We use a regex to remove ```json ... ``` or ``` ... ```
        fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
        # If fences exist, take the inner group
        match = fence_pattern.search(text)
        if match:
            return match.group(1)
        # Otherwise, return original text
        return text

    @classmethod
    def parse(cls, raw: str) -> dict:
        """
        Extract and parse the first JSON object found in `raw`. Uses dirty_json to tolerate minor errors.

        Args:
            raw: The raw string from the VLM (which may include markdown fences, prompts, etc.)

        Returns:
            A Python dict parsed from the first JSON object.

        Raises:
            ValueError if no JSON object can be extracted or parsed.
        """
        # 1) Strip out any markdown fences to simplify the search for '{'
        without_fences = cls._strip_markdown_fences(raw)

        # 2) Extract the first balanced { ... } block
        print(without_fences)
        try:
            json_block = cls._extract_braced_block(without_fences)
        except ValueError as e:
            raise ValueError(f"Failed to locate JSON block in VLM output: {e}")

        # 3) Use dirty_json to load into a dict (tolerant of trailing commas, unquoted keys, etc.)
        try:
            return dirtyjson.loads(json_block)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON block with dirty_json: {e}\nBlock was:\n{json_block!r}")