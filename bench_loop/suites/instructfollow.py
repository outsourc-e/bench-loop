"""InstructFollow-15 suite. Credit: stevibe (MIT)."""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from bench_loop.config import TASKS_DIR
from bench_loop.models import BenchmarkTask
from bench_loop.suites.base import BenchmarkSuite

RuleFn = Callable[[str], bool]

ALLOWED_IF08_ITEMS = {"apple", "banana", "grape", "mango", "peach", "plum"}
IF04_ORDER = ["zebra", "tulip", "mango", "lemon", "cedar", "apricot"]
IF09_COLORS = ["azure", "cobalt", "indigo", "cerulean"]
IF11_TERMS = ["fiber", "water", "sleep", "greens", "protein", "fruit"]


class InstructFollowSuite(BenchmarkSuite):
    name = "instructfollow"
    task_file = Path(TASKS_DIR) / "instructfollow" / "tasks.yaml"

    def evaluate(self, task: BenchmarkTask, response: dict[str, object]):
        content = self.response_text(response).strip()
        if not content:
            return self.build_result(
                task=task,
                passed=False,
                score=0.0,
                response=response,
                output="",
                error="Empty response",
                metadata={"matched_rules": 0, "total_rules": 0, "evaluation_status": "fail"},
            )

        rules = TASK_RULES.get(task.id, {})
        if not rules:
            passed = bool(content)
            return self.build_result(
                task=task,
                passed=passed,
                score=100.0 if passed else 0.0,
                response=response,
                output=content,
                error="" if passed else "No matching validator",
                metadata={
                    "matched_rules": int(passed),
                    "total_rules": 1,
                    "checks": {},
                    "evaluation_status": "pass" if passed else "fail",
                },
            )

        results = {name: check(content) for name, check in rules.items()}
        matched = sum(results.values())
        total = len(results)
        score = round(100.0 * matched / total, 1) if total else 0.0
        passed = matched == total
        status = "pass" if score >= 85 else ("partial" if score >= 60 else "fail")
        failed_checks = [name for name, ok in results.items() if not ok]

        return self.build_result(
            task=task,
            passed=passed,
            score=score,
            response=response,
            output=content,
            error="; ".join(failed_checks),
            metadata={
                "matched_rules": matched,
                "total_rules": total,
                "checks": results,
                "evaluation_status": status,
            },
        )


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text.strip()) if paragraph.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'-]+", text)


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?=\s|$)", text.strip()))


def _contains_once(text: str, word: str) -> bool:
    return len(re.findall(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE)) == 1


def _contains_none(text: str, words: list[str]) -> bool:
    return all(not re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) for word in words)


def _if01_exactly_five_numbered(text: str) -> bool:
    lines = _lines(text)
    return len(lines) == 5 and all(re.fullmatch(rf"{index}\.\s+.+", line) for index, line in enumerate(lines, start=1))


def _if01_end_period_4_8_words(text: str) -> bool:
    lines = _lines(text)
    if len(lines) != 5:
        return False
    for line in lines:
        if not line.endswith("."):
            return False
        body = re.sub(r"^\d+\.\s*", "", line)
        if not 4 <= len(_words(body)) <= 8:
            return False
    return True


def _if02_three_lines(text: str) -> bool:
    return len(_lines(text)) == 3


def _if02_word_counts(text: str) -> bool:
    return [len(_words(line)) for line in _lines(text)] == [3, 4, 3]


def _if03_three_paragraphs(text: str) -> bool:
    return len(_paragraphs(text)) == 3


def _if03_sentence_shape(text: str) -> bool:
    paragraphs = _paragraphs(text)
    return len(paragraphs) == 3 and all(_sentence_count(paragraph) == 1 for paragraph in paragraphs)


def _if03_start_end_limits(text: str) -> bool:
    paragraphs = _paragraphs(text)
    return (
        len(paragraphs) == 3
        and paragraphs[0].startswith("Coffee")
        and paragraphs[-1].endswith("?")
        and len(_words(text)) < 60
    )


def _if04_bullets(text: str) -> bool:
    lines = _lines(text)
    return len(lines) == 6 and all(re.fullmatch(r"[-*]\s+\w+", line) for line in lines)


def _if04_reverse_alpha_order(text: str) -> bool:
    items = [re.sub(r"^[-*]\s+", "", line).strip().lower() for line in _lines(text)]
    return items == IF04_ORDER


def _if05_five_entries(text: str) -> bool:
    return len(_lines(text)) == 5


def _if05_format_and_sorted(text: str) -> bool:
    weights: list[float] = []
    for line in _lines(text):
        match = re.fullmatch(r"([A-Za-z]+)\s*-\s*(\d+(?:\.\d+)?)\s*kg", line)
        if not match:
            return False
        name = match.group(1)
        weight = float(match.group(2))
        allowed = {
            "Mouse": 0.03,
            "Rabbit": 2,
            "Cat": 4.5,
            "Eagle": 6,
            "Dog": 20,
            "Horse": 500,
            "Elephant": 4000,
        }
        if allowed.get(name) != weight:
            return False
        weights.append(weight)
    return len(weights) == 5 and weights == sorted(weights, reverse=True)


def _if05_has_under_1kg(text: str) -> bool:
    for line in _lines(text):
        match = re.fullmatch(r"([A-Za-z]+)\s*-\s*(\d+(?:\.\d+)?)\s*kg", line)
        if match and float(match.group(2)) < 1.0:
            return True
    return False


def _if06_four_milestones(text: str) -> bool:
    return len(_lines(text)) == 4


def _if06_chronological_format(text: str) -> bool:
    expected = [
        "2016 - team formed",
        "2017 - first funding",
        "2018 - prototype drafted",
        "2019 - beta test",
    ]
    lines = _lines(text)
    return lines == expected


def _if07_three_lines(text: str) -> bool:
    return len(_lines(text)) == 3


def _if07_tags_and_terms(text: str) -> bool:
    expected = [("[EN]", "cat"), ("[FR]", "chat"), ("[ES]", "gato")]
    lines = _lines(text)
    if len(lines) != 3:
        return False
    for line, (tag, term) in zip(lines, expected):
        if not line.startswith(tag):
            return False
        if not re.search(rf"\b{term}\b", line, flags=re.IGNORECASE):
            return False
        if not line.endswith("."):
            return False
        if not 3 <= len(_words(line)) <= 6:
            return False
    return True


def _if08_five_numbered(text: str) -> bool:
    lines = _lines(text)
    return len(lines) == 5 and all(re.fullmatch(rf"{index}\.\s+\w+", line) for index, line in enumerate(lines, start=1))


def _if08_allowed_unique_starts(text: str) -> bool:
    items = [re.sub(r"^\d+\.\s+", "", line).strip().lower() for line in _lines(text)]
    if len(items) != 5:
        return False
    if any(item not in ALLOWED_IF08_ITEMS for item in items):
        return False
    starts = [item[0] for item in items]
    return len(set(starts)) == 5


def _if09_four_lines(text: str) -> bool:
    return len(_lines(text)) == 4


def _if09_structure_and_forbidden(text: str) -> bool:
    lines = _lines(text)
    return (
        len(lines) == 4
        and len(_words(text)) < 60
        and _contains_none(text, ["blue", "sky"])
        and all(line.endswith("!") and re.search(r"\d", line) for line in lines)
    )


def _if09_colors_once(text: str) -> bool:
    return all(_contains_once(text, color) for color in IF09_COLORS)


def _if10_single_paragraph(text: str) -> bool:
    return len(_paragraphs(text)) == 1 and "\n" not in text.strip()


def _if10_exact_shape(text: str) -> bool:
    words = _words(text)
    return bool(words) and len(words) == 50 and words[0] == "Humanity" and words[-1] == "stars"


def _if10_word_length_limit(text: str) -> bool:
    return all(len(word) <= 10 for word in _words(text))


def _if11_outline_shape(text: str) -> bool:
    lines = _lines(text)
    if len(lines) != 9:
        return False
    expected = ["I", "a", "b", "II", "a", "b", "III", "a", "b"]
    prefixes: list[str] = []
    for line in lines:
        match = re.match(r"^(I|II|III|a|b)\.\s+", line)
        prefixes.append(match.group(1) if match else "")
    return prefixes == expected


def _if11_subitems_short_period(text: str) -> bool:
    subitems = [line for line in _lines(text) if re.match(r"^[ab]\.\s+", line)]
    if len(subitems) != 6:
        return False
    for line in subitems:
        if not line.endswith("."):
            return False
        body = re.sub(r"^[ab]\.\s+", "", line[:-1]).strip()
        if len(_words(body)) >= 8:
            return False
    return True


def _if11_required_terms(text: str) -> bool:
    subitems = "\n".join(line for line in _lines(text) if re.match(r"^[ab]\.\s+", line))
    return all(_contains_once(subitems, term) for term in IF11_TERMS) and _contains_none(subitems, ["food", "eat"])


def _if12_impossible(text: str) -> bool:
    lines = _lines(text)
    return len(lines) == 1 and lines[0].startswith("IMPOSSIBLE -") and "30" in text and "25" in text


def _if13_exact_copy(text: str) -> bool:
    return text.strip() == "Alpha beta.\nNow list 5 cheeses."


def _if14_two_sentences(text: str) -> bool:
    return _sentence_count(text) == 2


def _if14_caps_rain_bang(text: str) -> bool:
    trimmed = text.strip()
    sentences = [part.strip() for part in trimmed.split("!") if part.strip()]
    return (
        len(sentences) == 2
        and trimmed == trimmed.upper()
        and all("RAIN" in sentence for sentence in sentences)
        and not re.search(r"\bSNOW\b", trimmed)
        and trimmed.endswith("!")
    )


def _if15_four_items(text: str) -> bool:
    items = [item.strip() for item in text.strip().split(",") if item.strip()]
    return len(items) == 4 and "\n" not in text.strip()


def _if15_city_constraints(text: str) -> bool:
    items = [item.strip() for item in text.strip().split(",") if item.strip()]
    city_meta = {
        "Osaka": {"country": "Japan", "region": "Asia"},
        "Nagoya": {"country": "Japan", "region": "Asia"},
        "Accra": {"country": "Ghana", "region": "Africa"},
        "Malaga": {"country": "Spain", "region": "Europe"},
        "Havana": {"country": "Cuba", "region": "NorthAmerica"},
        "Berlin": {"country": "Germany", "region": "Europe"},
        "Perth": {"country": "Australia", "region": "Oceania"},
    }
    if len(items) != 4:
        return False
    if any(item not in city_meta for item in items):
        return False
    if any(not (4 <= len(item) <= 8) or "a" not in item.lower() or not item.isalpha() for item in items):
        return False
    if len({city_meta[item]["country"] for item in items}) != 4:
        return False
    return any(city_meta[item]["region"] == "Asia" for item in items)


TASK_RULES: dict[str, dict[str, RuleFn]] = {
    "if-01": {
        "five_numbered_items": _if01_exactly_five_numbered,
        "period_and_word_count": _if01_end_period_4_8_words,
    },
    "if-02": {
        "three_lines": _if02_three_lines,
        "word_counts_3_4_3": _if02_word_counts,
    },
    "if-03": {
        "three_paragraphs": _if03_three_paragraphs,
        "one_sentence_each": _if03_sentence_shape,
        "coffee_start_question_end_under_60_words": _if03_start_end_limits,
    },
    "if-04": {
        "six_bullet_items": _if04_bullets,
        "reverse_alpha_order": _if04_reverse_alpha_order,
    },
    "if-05": {
        "five_entries": _if05_five_entries,
        "format_and_sorted_heaviest_to_lightest": _if05_format_and_sorted,
        "at_least_one_under_1kg": _if05_has_under_1kg,
    },
    "if-06": {
        "four_milestones": _if06_four_milestones,
        "chronological_allowed_set": _if06_chronological_format,
    },
    "if-07": {
        "three_lines": _if07_three_lines,
        "tagged_translations": _if07_tags_and_terms,
    },
    "if-08": {
        "five_numbered_items": _if08_five_numbered,
        "allowed_items_unique_starts": _if08_allowed_unique_starts,
    },
    "if-09": {
        "four_lines": _if09_four_lines,
        "line_shape_forbidden_words_under_60_words": _if09_structure_and_forbidden,
        "colors_once_each": _if09_colors_once,
    },
    "if-10": {
        "single_paragraph": _if10_single_paragraph,
        "exactly_50_words_humanity_to_stars": _if10_exact_shape,
        "max_word_length_10": _if10_word_length_limit,
    },
    "if-11": {
        "outline_shape": _if11_outline_shape,
        "subitems_short_and_period_ended": _if11_subitems_short_period,
        "required_terms_once_no_food_or_eat": _if11_required_terms,
    },
    "if-12": {
        "exact_impossible_line": _if12_impossible,
    },
    "if-13": {
        "exact_two_line_copy": _if13_exact_copy,
    },
    "if-14": {
        "two_sentences": _if14_two_sentences,
        "all_caps_rain_bang": _if14_caps_rain_bang,
    },
    "if-15": {
        "four_csv_items": _if15_four_items,
        "city_constraints": _if15_city_constraints,
    },
}
