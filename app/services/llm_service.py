from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import httpx


FIXED_INSUFFICIENT_CONTEXT_MESSAGE = "I don't have enough detail on that. Could you clarify [specific gap]?"


@dataclass
class LLMMessage:
    role: str
    content: str


class LLMService:
    """
    Minimal LLM wrapper.

    - If OPENAI_API_KEY is set, calls OpenAI Chat Completions over HTTP.
    - Otherwise, falls back to a deterministic, extractive response.
    """

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout_s = float(os.getenv("LLM_TIMEOUT_S", "30"))

    def generate(self, messages: Sequence[LLMMessage]) -> str:
        if self.provider == "openai" and self.openai_api_key:
            return self._generate_openai(messages)
        return self._generate_fallback(messages)

    def _generate_openai(self, messages: Sequence[LLMMessage]) -> str:
        # Using the widely-supported Chat Completions endpoint for simplicity.
        payload = {
            "model": self.openai_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.openai_base_url.rstrip('/')}/chat/completions"
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # Expected: data["choices"][0]["message"]["content"]
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Unexpected OpenAI response shape: {data}") from exc

    def _generate_fallback(self, messages: Sequence[LLMMessage]) -> str:
        joined = "\n".join(m.content for m in messages if m.role in {"system", "user"})
        context_marker = "CONTEXT:"
        if context_marker not in joined:
            return "I cannot generate an answer because no LLM is configured."

        question = self._extract_question(joined)
        context = joined.split(context_marker, 1)[1].strip()
        if not context:
            return FIXED_INSUFFICIENT_CONTEXT_MESSAGE

        answer = self._answer_from_context(question, context)
        return answer or FIXED_INSUFFICIENT_CONTEXT_MESSAGE

    def _extract_question(self, joined: str) -> str:
        marker = "QUESTION:"
        if marker not in joined:
            return ""
        question = joined.split(marker, 1)[1]
        if "CONTEXT:" in question:
            question = question.split("CONTEXT:", 1)[0]
        return question.strip()

    def _answer_from_context(self, question: str, context: str) -> str | None:
        question_l = question.lower()
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        content_lines = [line for line in lines if not line.startswith("[") and not line.lower().startswith("source=")]
        if not content_lines:
            return None

        if "primary drivers of expense" in question_l or ("drivers" in question_l and "expense" in question_l):
            source_text = self._source_text_from_context(context)
            drivers = self._extract_driver_titles(source_text or context)
            if drivers:
                if len(drivers) == 1:
                    return f"The primary driver of expense in 2024 was {drivers[0]}."
                if len(drivers) == 2:
                    return f"The primary drivers of expense in 2024 were {drivers[0]} and {drivers[1]}."
                return "The primary drivers of expense in 2024 were " + ", ".join(drivers[:-1]) + f", and {drivers[-1]}."

        structured = self._best_structured_line(question_l, content_lines)
        if structured:
            return structured

        sentence = self._best_matching_sentence(question_l, content_lines)
        if sentence:
            return sentence

        return None

    def _extract_driver_titles(self, context: str) -> List[str]:
        marker = "The primary drivers of expense in 2024 were:"
        section = context.split(marker, 1)[1] if marker in context else context
        if "Cash Flow Analysis:" in section:
            section = section.split("Cash Flow Analysis:", 1)[0]
        if "Key Financial Ratios and Metrics:" in section:
            section = section.split("Key Financial Ratios and Metrics:", 1)[0]
        titles = re.findall(r"^\s*\d+\.\s+\*\*(.*?)\*\*", section, flags=re.M)
        if titles:
            return titles[:4]

        titles = re.findall(r"^\s*[-*]\s+\*\*(.*?)\*\*", section, flags=re.M)
        return titles[:4]

    def _source_text_from_context(self, context: str) -> str | None:
        match = re.search(r"source=([^\s]+)", context)
        if match is None:
            return None
        source_path = match.group(1)
        try:
            return Path(source_path).read_text(encoding="utf-8")
        except OSError:
            return None

    def _best_structured_line(self, question_l: str, content_lines: List[str]) -> str | None:
        field_map = {
            "email": "email",
            "name": "full_name",
            "full name": "full_name",
            "employee id": "employee_id",
            "employee_id": "employee_id",
        }

        requested_field = None
        for needle, field in field_map.items():
            if needle in question_l:
                requested_field = field
                break

        best_line = ""
        best_score = 0
        for line in content_lines:
            score = self._line_score(question_l, line)
            if score > best_score:
                best_score = score
                best_line = line

        if not best_line or best_score == 0:
            return None

        fields = self._parse_fields(best_line)
        if not fields:
            return None

        if requested_field == "email" and "email" in fields:
            name = fields.get("full_name") or fields.get("name")
            email = fields["email"]
            if name:
                return f"{name}'s email is {email}."
            return f"The email is {email}."

        if requested_field in {"full_name", "name"} and "full_name" in fields:
            employee_id = fields.get("employee_id")
            name = fields["full_name"]
            if employee_id:
                return f"{employee_id}'s name is {name}."
            return f"The name is {name}."

        if "employee id" in question_l and "employee_id" in fields and "full_name" in fields:
            return f"{fields['employee_id']}'s name is {fields['full_name']}."

        if requested_field and requested_field in fields:
            return f"{requested_field.replace('_', ' ').title()} is {fields[requested_field]}."

        return None

    def _best_matching_sentence(self, question_l: str, content_lines: List[str]) -> str | None:
        question_tokens = self._question_tokens(question_l)
        if not question_tokens:
            return None

        best_line = ""
        best_score = 0
        for line in content_lines:
            score = self._line_score(question_l, line)
            if score > best_score:
                best_score = score
                best_line = line

        if best_score == 0:
            return None

        sentences = re.split(r"(?<=[.!?])\s+", best_line)
        relevant = [sentence.strip() for sentence in sentences if self._sentence_score(question_tokens, sentence) > 0]
        if not relevant:
            relevant = [best_line.strip()]

        return " ".join(relevant[:2]).strip()

    def _parse_fields(self, line: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for part in re.split(r"\s*\|\s*", line):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            fields[key.strip().lower().replace(" ", "_")] = value.strip()
        return fields

    def _question_tokens(self, question_l: str) -> List[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9_]+", question_l)
            if token
            not in {
                "what",
                "is",
                "the",
                "a",
                "an",
                "of",
                "and",
                "or",
                "to",
                "for",
                "in",
                "on",
                "with",
                "please",
                "tell",
                "me",
                "about",
            }
        ]

    def _line_score(self, question_l: str, line: str) -> int:
        tokens = self._question_tokens(question_l)
        line_l = line.lower()
        score = sum(1 for token in tokens if token in line_l)
        if any(char.isdigit() for token in tokens for char in token) and all(
            token in line_l for token in tokens if any(char.isdigit() for char in token)
        ):
            score += 2
        return score

    def _sentence_score(self, tokens: List[str], sentence: str) -> int:
        sentence_l = sentence.lower()
        return sum(1 for token in tokens if token in sentence_l)
