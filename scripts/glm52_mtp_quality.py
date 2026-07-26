#!/usr/bin/env python3
"""Fail-closed matched MTP quality evidence for the GLM-5.2 Marlin lane.

The module intentionally owns no server lifecycle and downloads no datasets.
Callers provide already-tokenized cases and one bounded native ``/generate``
callback.  This keeps the first representative gate cheap while giving both
the MTP-off and MTP-on runs exactly the same request and response contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final, Literal, cast, final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))

from scripts.glm52_routing_profiles import (  # noqa: E402
    RepresentativePrompt,
    representative_prompt_manifest,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type QualityCaseCategory = Literal["coding", "agent", "humaneval"]
type TeacherForcedTargetKind = Literal["prompt_self_likelihood", "reference_completion"]
type NativeGenerate = Callable[[JsonObject], JsonObject]
type PromptEncoder = Callable[[str], Sequence[int]]

GLM52_VOCABULARY_SIZE: Final = 154_880
DEFAULT_SAMPLING_SEED: Final = 20_260_725
MAXIMUM_GENERATION_TOKENS: Final = 512
MAXIMUM_TOP_LOGPROBS: Final = 64
MARLIN_LOCAL_HEADS_PER_MODULE: Final = 32
TARGET_LAYER_COUNT: Final = 78
MTP_LAYER_COUNT: Final = 1
MAXIMUM_VALIDATION_ERROR_BYTES: Final = 2_048
MAXIMUM_VALIDATION_ERROR_ITEMS: Final = 8
_SHA256_LENGTH: Final = 64

HUMANEVAL_SUBSET_IDS: Final[tuple[str, ...]] = (
    "HumanEval/0",
    "HumanEval/2",
    "HumanEval/11",
    "HumanEval/28",
    "HumanEval/32",
    "HumanEval/53",
    "HumanEval/72",
    "HumanEval/149",
)

# The routing-profile module currently publishes the corpus manifest, but not
# the prompt bodies.  Keep this immutable mirror bound to that public manifest
# so drift fails at import instead of silently changing one side of a pair.
REPRESENTATIVE_PROMPTS: Final[tuple[RepresentativePrompt, ...]] = (
    RepresentativePrompt(
        prompt_id="coding-python-race",
        category="coding",
        text=(
            "You are reviewing an asyncio Python service. A producer appends work "
            "items to a deque, sets an Event, and a consumer clears the Event after "
            "draining the deque. Under load, an item can remain queued forever. "
            "Explain the lost-wakeup interleaving, propose the smallest exact fix, "
            "and give a focused pytest-asyncio regression test. Preserve cancellation "
            "and do not replace the queue with polling."
        ),
    ),
    RepresentativePrompt(
        prompt_id="coding-rust-protocol",
        category="coding",
        text=(
            "Design a backwards-compatible Rust wire-protocol change that adds a "
            "content hash and monotonic generation to an artifact message. The "
            "decoder must reject ambiguous legacy/new encodings, integer overflow, "
            "duplicate fields, and trailing bytes. Show the typed data model, parsing "
            "invariants, and property tests; avoid unwrap in the network boundary."
        ),
    ),
    RepresentativePrompt(
        prompt_id="coding-cuda-overlap",
        category="coding",
        text=(
            "A CUDA inference layer stages activations to pinned host memory, starts "
            "CPU expert work, computes a GPU branch, and then merges outputs. Two "
            "requests may overlap. Review the ownership protocol needed for a shared "
            "staging buffer: leases, generations, stream events, exception cleanup, "
            "and stale-handle rejection. Give pseudocode and identify deadlocks."
        ),
    ),
    RepresentativePrompt(
        prompt_id="coding-sql-migration",
        category="coding",
        text=(
            "Plan an online PostgreSQL migration from a nullable text identifier to "
            "a non-null UUID primary key for a high-write table. Include shadow "
            "columns, deterministic backfill, dual writes, validation, index build, "
            "cutover, rollback, and observability. State which operations can lock "
            "and how an agent should prove each phase before advancing."
        ),
    ),
    RepresentativePrompt(
        prompt_id="agent-incident",
        category="agent",
        text=(
            "Act as an infrastructure agent investigating intermittent distributed "
            "inference stalls. Evidence: GPU utilization alternates between 0 and "
            "95 percent, CPU memory bandwidth stays high, InfiniBand counters are "
            "clean, and only concurrency two stalls. Produce a ranked hypothesis "
            "tree, exact read-only checks, stopping conditions, and a minimal "
            "experiment that distinguishes scheduling serialization from transport."
        ),
    ),
    RepresentativePrompt(
        prompt_id="agent-repository",
        category="agent",
        text=(
            "You inherit a dirty repository with unrelated user edits and a failing "
            "CI type check. Implement a narrowly scoped feature that touches Python "
            "and Rust without losing work. Describe how you inspect ownership, split "
            "parallel tasks, preserve the worktree, validate only relevant blockers, "
            "and produce a reviewable handoff with exact files and evidence."
        ),
    ),
    RepresentativePrompt(
        prompt_id="agent-deployment",
        category="agent",
        text=(
            "Create an immutable two-host model deployment plan over five network "
            "paths with heterogeneous rates. Artifacts may land on disk, tmpfs, or "
            "remain streamed when capacity is insufficient. Require content hashes, "
            "resume safety, link identity checks, failover, cache eviction rules, and "
            "a receipt that proves which bytes each host consumed."
        ),
    ),
    RepresentativePrompt(
        prompt_id="agent-benchmark",
        category="agent",
        text=(
            "Design a realistic language-model benchmark for concurrency one and two "
            "that separates prompt processing, decode-window throughput, and "
            "end-to-end throughput. Reuse a semantic warm-up without flushing model "
            "state, pin prompts and sampling, detect serialized admission, and "
            "explain why rolling decode logs can exceed the final aggregate rate."
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class HumanEvalReference:
    task_id: str
    prompt: str
    canonical_solution: str


HUMANEVAL_REFERENCES: Final[tuple[HumanEvalReference, ...]] = (
    HumanEvalReference(
        task_id="HumanEval/0",
        prompt=(
            "from typing import List\n\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """ Check if in given list of numbers, are any two numbers closer '
            "to each other than\n"
            "    given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            '    """\n'
        ),
        canonical_solution=(
            "    for idx, elem in enumerate(numbers):\n"
            "        for idx2, elem2 in enumerate(numbers):\n"
            "            if idx != idx2:\n"
            "                distance = abs(elem - elem2)\n"
            "                if distance < threshold:\n"
            "                    return True\n\n"
            "    return False\n"
        ),
    ),
    HumanEvalReference(
        task_id="HumanEval/2",
        prompt=(
            "\n\ndef truncate_number(number: float) -> float:\n"
            '    """ Given a positive floating point number, it can be decomposed '
            "into\n"
            "    and integer part (largest integer smaller than given number) and "
            "decimals\n"
            "    (leftover part always smaller than 1).\n\n"
            "    Return the decimal part of the number.\n"
            "    >>> truncate_number(3.5)\n"
            "    0.5\n"
            '    """\n'
        ),
        canonical_solution="    return number % 1.0\n",
    ),
    HumanEvalReference(
        task_id="HumanEval/11",
        prompt=(
            "from typing import List\n\n\n"
            "def string_xor(a: str, b: str) -> str:\n"
            '    """ Input are two strings a and b consisting only of 1s and 0s.\n'
            "    Perform binary XOR on these inputs and return result also as a "
            "string.\n"
            "    >>> string_xor('010', '110')\n"
            "    '100'\n"
            '    """\n'
        ),
        canonical_solution=(
            "    def xor(i, j):\n"
            "        if i == j:\n"
            "            return '0'\n"
            "        else:\n"
            "            return '1'\n\n"
            "    return ''.join(xor(x, y) for x, y in zip(a, b))\n"
        ),
    ),
    HumanEvalReference(
        task_id="HumanEval/28",
        prompt=(
            "from typing import List\n\n\n"
            "def concatenate(strings: List[str]) -> str:\n"
            '    """ Concatenate list of strings into a single string\n'
            "    >>> concatenate([])\n"
            "    ''\n"
            "    >>> concatenate(['a', 'b', 'c'])\n"
            "    'abc'\n"
            '    """\n'
        ),
        canonical_solution="    return ''.join(strings)\n",
    ),
    HumanEvalReference(
        task_id="HumanEval/32",
        prompt=(
            "import math\n\n\n"
            "def poly(xs: list, x: float):\n"
            '    """\n'
            "    Evaluates polynomial with coefficients xs at point x.\n"
            "    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n\n"
            '    """\n'
            "    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])\n"
            "\n\ndef find_zero(xs: list):\n"
            '    """ xs are coefficients of a polynomial.\n'
            "    find_zero find x such that poly(x) = 0.\n"
            "    find_zero returns only only zero point, even if there are many.\n"
            "    Moreover, find_zero only takes list xs having even number of "
            "coefficients\n"
            "    and largest non zero coefficient as it guarantees\n"
            "    a solution.\n"
            "    >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x\n"
            "    -0.5\n"
            "    >>> round(find_zero([-6, 11, -6, 1]), 2) # (x - 1) * "
            "(x - 2) * (x - 3) = -6 + 11x - 6x^2 + x^3\n"
            "    1.0\n"
            '    """\n'
        ),
        canonical_solution=(
            "    begin, end = -1., 1.\n"
            "    while poly(xs, begin) * poly(xs, end) > 0:\n"
            "        begin *= 2.0\n"
            "        end *= 2.0\n"
            "    while end - begin > 1e-10:\n"
            "        center = (begin + end) / 2.0\n"
            "        if poly(xs, center) * poly(xs, begin) > 0:\n"
            "            begin = center\n"
            "        else:\n"
            "            end = center\n"
            "    return begin\n"
        ),
    ),
    HumanEvalReference(
        task_id="HumanEval/53",
        prompt=(
            '\n\ndef add(x: int, y: int):\n    """Add two numbers x and y\n'
            "    >>> add(2, 3)\n"
            "    5\n"
            "    >>> add(5, 7)\n"
            "    12\n"
            '    """\n'
        ),
        canonical_solution="    return x + y\n",
    ),
    HumanEvalReference(
        task_id="HumanEval/72",
        prompt=(
            "\ndef will_it_fly(q,w):\n"
            "    '''\n"
            "    Write a function that returns True if the object q will fly, and "
            "False otherwise.\n"
            "    The object q will fly if it's balanced (it is a palindromic list) "
            "and the sum of its elements is less than or equal the maximum possible "
            "weight w.\n\n"
            "    Example:\n"
            "    will_it_fly([1, 2], 5) ➞ False \n"
            "    # 1+2 is less than the maximum possible weight, but it's "
            "unbalanced.\n\n"
            "    will_it_fly([3, 2, 3], 1) ➞ False\n"
            "    # it's balanced, but 3+2+3 is more than the maximum possible "
            "weight.\n\n"
            "    will_it_fly([3, 2, 3], 9) ➞ True\n"
            "    # 3+2+3 is less than the maximum possible weight, and it's "
            "balanced.\n\n"
            "    will_it_fly([3], 5) ➞ True\n"
            "    # 3 is less than the maximum possible weight, and it's balanced.\n"
            "    '''\n"
        ),
        canonical_solution=(
            "    if sum(q) > w:\n"
            "        return False\n\n"
            "    i, j = 0, len(q)-1\n"
            "    while i<j:\n"
            "        if q[i] != q[j]:\n"
            "            return False\n"
            "        i+=1\n"
            "        j-=1\n"
            "    return True\n"
        ),
    ),
    HumanEvalReference(
        task_id="HumanEval/149",
        prompt=(
            '\ndef sorted_list_sum(lst):\n    """Write a function that accepts '
            "a list of strings as a parameter,\n"
            "    deletes the strings that have odd lengths from it,\n"
            "    and returns the resulted list with a sorted order,\n"
            "    The list is always a list of strings and never an array of "
            "numbers,\n"
            "    and it may contain duplicates.\n"
            "    The order of the list should be ascending by length of each word, "
            "and you\n"
            "    should return the list sorted by that rule.\n"
            "    If two words have the same length, sort the list alphabetically.\n"
            "    The function should return a list of strings in sorted order.\n"
            "    You may assume that all words will have the same length.\n"
            "    For example:\n"
            '    assert list_sort(["aa", "a", "aaa"]) => ["aa"]\n'
            '    assert list_sort(["ab", "a", "aaa", "cd"]) => ["ab", "cd"]\n'
            '    """\n'
        ),
        canonical_solution=(
            "    lst.sort()\n"
            "    new_lst = []\n"
            "    for i in lst:\n"
            "        if len(i)%2 == 0:\n"
            "            new_lst.append(i)\n"
            "    return sorted(new_lst, key=len)\n"
        ),
    ),
)

if tuple(reference.task_id for reference in HUMANEVAL_REFERENCES) != (
    HUMANEVAL_SUBSET_IDS
):
    raise RuntimeError("embedded HumanEval rows do not match the frozen subset")

HUMANEVAL_REFERENCE_CONTENT_SHA256: Final = hashlib.sha256(
    json.dumps(
        [asdict(reference) for reference in HUMANEVAL_REFERENCES],
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
).hexdigest()


class Glm52MtpQualityError(RuntimeError):
    """Raised when quality evidence is incomplete, ambiguous, or mismatched."""


def _canonical_json_bytes(value: JsonValue) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise Glm52MtpQualityError("value is not canonical finite JSON") from error


def _canonical_sha256(value: JsonValue) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_sha256(value: str, description: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")


def _validate_case_id(case_id: str) -> None:
    if not case_id or len(case_id) > 128 or case_id.strip() != case_id:
        raise ValueError("quality case ID is invalid")


def _validate_token_ids(token_ids: tuple[int, ...], *, minimum_count: int) -> None:
    if len(token_ids) < minimum_count or any(
        token_id < 0 or token_id >= GLM52_VOCABULARY_SIZE for token_id in token_ids
    ):
        raise ValueError("token IDs are outside the GLM-5.2 contract")


def token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    _validate_token_ids(token_ids, minimum_count=1)
    return _canonical_sha256(list(token_ids))


def representative_prompts() -> tuple[RepresentativePrompt, ...]:
    """Return the immutable eight-prompt coding/agent corpus."""

    return REPRESENTATIVE_PROMPTS


def humaneval_references() -> tuple[HumanEvalReference, ...]:
    """Return the embedded prompt/reference rows for the frozen task IDs."""

    return HUMANEVAL_REFERENCES


def _validate_representative_prompt_binding() -> None:
    manifest = representative_prompt_manifest()
    raw_prompts = manifest.get("prompts")
    if not isinstance(raw_prompts, list) or len(raw_prompts) != len(
        REPRESENTATIVE_PROMPTS
    ):
        raise Glm52MtpQualityError("representative prompt manifest count drifted")
    expected: list[JsonValue] = [
        {
            "prompt_id": prompt.prompt_id,
            "category": prompt.category,
            "utf8_bytes": len(prompt.text.encode()),
            "text_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
        }
        for prompt in REPRESENTATIVE_PROMPTS
    ]
    if raw_prompts != expected:
        raise Glm52MtpQualityError("representative prompt manifest content drifted")


_validate_representative_prompt_binding()


@dataclass(frozen=True, slots=True)
class QualityProfile:
    """A frozen set of cases and request sizes for one matched pair."""

    name: str
    representative_prompt_ids: tuple[str, ...]
    humaneval_ids: tuple[str, ...]
    generation_max_new_tokens: int
    top_k: int
    sampling_seed: int = DEFAULT_SAMPLING_SEED

    def __post_init__(self) -> None:
        all_case_ids = self.case_ids
        if (
            not self.name
            or not all_case_ids
            or len(set(all_case_ids)) != len(all_case_ids)
            or any(not case_id for case_id in all_case_ids)
            or not 1 <= self.generation_max_new_tokens <= MAXIMUM_GENERATION_TOKENS
            or not 1 <= self.top_k <= MAXIMUM_TOP_LOGPROBS
            or not 0 <= self.sampling_seed < 2**31
        ):
            raise ValueError("quality profile is invalid")
        known_prompt_ids = {prompt.prompt_id for prompt in REPRESENTATIVE_PROMPTS}
        if not set(self.representative_prompt_ids) <= known_prompt_ids:
            raise ValueError("quality profile has an unknown representative prompt")
        if not set(self.humaneval_ids) <= set(HUMANEVAL_SUBSET_IDS):
            raise ValueError("quality profile has an unfrozen HumanEval task")

    @property
    def case_ids(self) -> tuple[str, ...]:
        return self.representative_prompt_ids + self.humaneval_ids

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(cast(JsonValue, asdict(self)))


_REPRESENTATIVE_PROMPT_IDS: Final = tuple(
    prompt.prompt_id for prompt in REPRESENTATIVE_PROMPTS
)

THIN_FIRST_RUN_PROFILE: Final = QualityProfile(
    name="glm52-marlin-mtp-thin-v1",
    representative_prompt_ids=_REPRESENTATIVE_PROMPT_IDS,
    humaneval_ids=HUMANEVAL_SUBSET_IDS[:2],
    generation_max_new_tokens=32,
    top_k=8,
)

REPRESENTATIVE_QUALITY_GATE_PROFILE: Final = QualityProfile(
    name="glm52-marlin-mtp-representative-v1",
    representative_prompt_ids=_REPRESENTATIVE_PROMPT_IDS,
    humaneval_ids=HUMANEVAL_SUBSET_IDS,
    generation_max_new_tokens=128,
    top_k=16,
)


@dataclass(frozen=True, slots=True)
class TokenizedQualityCase:
    case_id: str
    category: QualityCaseCategory
    input_ids: tuple[int, ...]
    teacher_forced_input_ids: tuple[int, ...] | None = None
    teacher_forced_score_start_index: int = 1
    teacher_forced_target_kind: TeacherForcedTargetKind = "prompt_self_likelihood"

    def __post_init__(self) -> None:
        _validate_case_id(self.case_id)
        _validate_token_ids(self.input_ids, minimum_count=2)
        has_explicit_teacher_sequence = self.teacher_forced_input_ids is not None
        teacher_forced_input_ids = self.teacher_forced_input_ids
        if teacher_forced_input_ids is None:
            teacher_forced_input_ids = self.input_ids
            object.__setattr__(
                self,
                "teacher_forced_input_ids",
                teacher_forced_input_ids,
            )
        _validate_token_ids(teacher_forced_input_ids, minimum_count=2)
        if (
            not 1
            <= self.teacher_forced_score_start_index
            < len(teacher_forced_input_ids)
        ):
            raise ValueError("teacher-forced scoring boundary is invalid")
        if (
            self.teacher_forced_target_kind == "reference_completion"
            and not has_explicit_teacher_sequence
        ):
            raise ValueError(
                "reference-completion scoring requires an explicit combined sequence"
            )
        if self.category == "humaneval" and self.case_id not in HUMANEVAL_SUBSET_IDS:
            raise ValueError("HumanEval case is not in the frozen subset")
        if self.category != "humaneval":
            categories = {
                prompt.prompt_id: prompt.category for prompt in REPRESENTATIVE_PROMPTS
            }
            if categories.get(self.case_id) != self.category:
                raise ValueError("representative quality case category is inconsistent")


def _encode_bounded(
    encoder: PromptEncoder,
    text: str,
    *,
    maximum_input_tokens: int,
) -> tuple[int, ...]:
    raw_token_ids = encoder(text)
    if isinstance(raw_token_ids, (str, bytes)) or not all(
        type(token_id) is int for token_id in raw_token_ids
    ):
        raise Glm52MtpQualityError("quality encoder returned invalid token IDs")
    token_ids = tuple(cast(Sequence[int], raw_token_ids))
    _validate_token_ids(token_ids, minimum_count=2)
    if len(token_ids) > maximum_input_tokens:
        raise Glm52MtpQualityError(
            f"quality input has {len(token_ids)} tokens; "
            f"limit is {maximum_input_tokens}"
        )
    return token_ids


def _common_prefix_length(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> int:
    length = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        length += 1
    return length


def tokenize_quality_cases(
    representative_prompt_encoder: PromptEncoder,
    humaneval_encoder: PromptEncoder,
    *,
    profile: QualityProfile = THIN_FIRST_RUN_PROFILE,
    maximum_input_tokens: int = 4_096,
) -> tuple[TokenizedQualityCase, ...]:
    """Tokenize the embedded cases without consulting a network dataset.

    The representative cases score prompt self-likelihood from token index one.
    HumanEval cases use the raw prompt for greedy generation and score only the
    canonical-reference region of the independently tokenized prompt+solution.
    The exact score boundary is the longest token prefix shared by the raw
    prompt and combined sequence, so a tokenizer merge at the text boundary is
    included in the reference region instead of being silently skipped.
    """

    if maximum_input_tokens < 2:
        raise ValueError("maximum input tokens must be at least two")
    representative_by_id = {
        prompt.prompt_id: prompt for prompt in REPRESENTATIVE_PROMPTS
    }
    humaneval_by_id = {
        reference.task_id: reference for reference in HUMANEVAL_REFERENCES
    }
    result: list[TokenizedQualityCase] = []
    for case_id in profile.representative_prompt_ids:
        prompt = representative_by_id[case_id]
        input_ids = _encode_bounded(
            representative_prompt_encoder,
            prompt.text,
            maximum_input_tokens=maximum_input_tokens,
        )
        result.append(
            TokenizedQualityCase(
                case_id=case_id,
                category=prompt.category,
                input_ids=input_ids,
            )
        )
    for case_id in profile.humaneval_ids:
        reference = humaneval_by_id[case_id]
        prompt_input_ids = _encode_bounded(
            humaneval_encoder,
            reference.prompt,
            maximum_input_tokens=maximum_input_tokens,
        )
        combined_input_ids = _encode_bounded(
            humaneval_encoder,
            reference.prompt + reference.canonical_solution,
            maximum_input_tokens=maximum_input_tokens,
        )
        score_start_index = _common_prefix_length(prompt_input_ids, combined_input_ids)
        if not 1 <= score_start_index < len(combined_input_ids):
            raise Glm52MtpQualityError(
                f"cannot bind HumanEval reference token boundary for {case_id}"
            )
        result.append(
            TokenizedQualityCase(
                case_id=case_id,
                category="humaneval",
                input_ids=prompt_input_ids,
                teacher_forced_input_ids=combined_input_ids,
                teacher_forced_score_start_index=score_start_index,
                teacher_forced_target_kind="reference_completion",
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TeacherForcedRequest:
    case_id: str
    input_ids: tuple[int, ...]
    score_start_index: int = 1
    target_kind: TeacherForcedTargetKind = "prompt_self_likelihood"
    sampling_seed: int = DEFAULT_SAMPLING_SEED

    def __post_init__(self) -> None:
        _validate_case_id(self.case_id)
        _validate_token_ids(self.input_ids, minimum_count=2)
        if (
            not 1 <= self.score_start_index < len(self.input_ids)
            or not 0 <= self.sampling_seed < 2**31
        ):
            raise ValueError("teacher-forced request is outside its bound")

    def json_object(self) -> JsonObject:
        return {
            "input_ids": list(self.input_ids),
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
                "ignore_eos": True,
                "sampling_seed": self.sampling_seed,
            },
            "stream": False,
            "return_logprob": True,
            # Installed SGLang emits a leading null logprob sentinel for regular
            # input scoring. Start one token earlier so every token in the
            # intended score region still has a finite teacher-forced logprob.
            "logprob_start_len": self.score_start_index - 1,
            "top_logprobs_num": 0,
            "return_text_in_logprobs": False,
            "log_metrics": False,
        }


@dataclass(frozen=True, slots=True)
class TopKProbeRequest:
    case_id: str
    input_ids: tuple[int, ...]
    top_k: int
    sampling_seed: int = DEFAULT_SAMPLING_SEED

    def __post_init__(self) -> None:
        _validate_case_id(self.case_id)
        _validate_token_ids(self.input_ids, minimum_count=2)
        if (
            not 1 <= self.top_k <= MAXIMUM_TOP_LOGPROBS
            or not 0 <= self.sampling_seed < 2**31
        ):
            raise ValueError("top-k probe request is outside its bound")

    def json_object(self) -> JsonObject:
        return {
            "input_ids": list(self.input_ids),
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
                "ignore_eos": True,
                "sampling_seed": self.sampling_seed,
            },
            "stream": False,
            "return_logprob": True,
            "logprob_start_len": -1,
            "top_logprobs_num": self.top_k,
            "return_text_in_logprobs": False,
            "log_metrics": False,
        }


@dataclass(frozen=True, slots=True)
class DeterministicGenerationRequest:
    case_id: str
    input_ids: tuple[int, ...]
    max_new_tokens: int
    sampling_seed: int = DEFAULT_SAMPLING_SEED

    def __post_init__(self) -> None:
        _validate_case_id(self.case_id)
        _validate_token_ids(self.input_ids, minimum_count=2)
        if (
            not 1 <= self.max_new_tokens <= MAXIMUM_GENERATION_TOKENS
            or not 0 <= self.sampling_seed < 2**31
        ):
            raise ValueError("generation request is outside its bound")

    def json_object(self) -> JsonObject:
        return {
            "input_ids": list(self.input_ids),
            "sampling_params": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "sampling_seed": self.sampling_seed,
            },
            "stream": False,
            "return_logprob": False,
            "log_metrics": False,
        }


class _PermissiveModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)


@final
class _LengthFinishReason(_PermissiveModel):
    type: Literal["length"]
    length: int = Field(gt=0)


type _SerializedLogprob = tuple[float, int, None]
type _SerializedInputLogprob = tuple[float | None, int, None]


@final
class _TeacherForcedMeta(_PermissiveModel):
    prompt_tokens: int = Field(gt=1)
    completion_tokens: Literal[1]
    cached_tokens: int = Field(ge=0)
    finish_reason: _LengthFinishReason
    input_token_logprobs: tuple[_SerializedInputLogprob, ...]
    output_token_logprobs: tuple[_SerializedLogprob, ...]


@final
class _TeacherForcedResponse(_PermissiveModel):
    text: str
    output_ids: tuple[int, ...]
    meta_info: _TeacherForcedMeta


@final
class _TopKProbeMeta(_PermissiveModel):
    prompt_tokens: int = Field(gt=1)
    completion_tokens: Literal[1]
    cached_tokens: int = Field(ge=0)
    finish_reason: _LengthFinishReason
    output_token_logprobs: tuple[_SerializedLogprob, ...]
    output_top_logprobs: tuple[tuple[_SerializedLogprob, ...], ...]


@final
class _TopKProbeResponse(_PermissiveModel):
    text: str
    output_ids: tuple[int, ...]
    meta_info: _TopKProbeMeta


@final
class _GenerationMeta(_PermissiveModel):
    prompt_tokens: int = Field(gt=1)
    completion_tokens: int = Field(gt=0)
    cached_tokens: int = Field(ge=0)
    finish_reason: _LengthFinishReason


@final
class _GenerationResponse(_PermissiveModel):
    text: str
    output_ids: tuple[int, ...]
    meta_info: _GenerationMeta


def _validated_response[ResponseModel: BaseModel](
    payload: JsonObject,
    model: type[ResponseModel],
    description: str,
) -> tuple[ResponseModel, str]:
    encoded = _canonical_json_bytes(payload)
    try:
        parsed = model.model_validate_json(encoded)
    except ValidationError as error:
        raw_errors = error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        validation_errors: list[JsonValue] = []
        for raw_error in raw_errors[:MAXIMUM_VALIDATION_ERROR_ITEMS]:
            raw_location = raw_error.get("loc", ())
            location: list[JsonValue] = [
                item if isinstance(item, (str, int)) else str(item)
                for item in raw_location
            ]
            validation_errors.append(
                {
                    "location": location,
                    "type": str(raw_error.get("type", "unknown"))[:128],
                    "message": str(raw_error.get("msg", "validation failed"))[:256],
                }
            )
        details: JsonObject = {
            "error_count": len(raw_errors),
            "errors": validation_errors,
            "truncated": len(raw_errors) > len(validation_errors),
        }
        details_json = _canonical_json_bytes(details).decode()
        if len(details_json.encode()) > MAXIMUM_VALIDATION_ERROR_BYTES:
            details = {
                "error_count": len(raw_errors),
                "errors": [],
                "truncated": True,
            }
            details_json = _canonical_json_bytes(details).decode()
        raise Glm52MtpQualityError(
            f"{description} is invalid: {details_json}"
        ) from error
    return parsed, hashlib.sha256(encoded).hexdigest()


def _validate_serialized_logprob(entry: _SerializedLogprob) -> None:
    logprob, token_id, text = entry
    if (
        not math.isfinite(logprob)
        or token_id < 0
        or token_id >= GLM52_VOCABULARY_SIZE
        or text is not None
    ):
        raise Glm52MtpQualityError("response contains an invalid token logprob")


def _validate_input_logprob_sentinel(entry: _SerializedInputLogprob) -> None:
    logprob, token_id, text = entry
    if (
        logprob is not None
        or token_id < 0
        or token_id >= GLM52_VOCABULARY_SIZE
        or text is not None
    ):
        raise Glm52MtpQualityError(
            "teacher-forced response lacks its exact leading null sentinel"
        )


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    token_id: int
    logprob: float

    def __post_init__(self) -> None:
        if (
            self.token_id < 0
            or self.token_id >= GLM52_VOCABULARY_SIZE
            or not math.isfinite(self.logprob)
        ):
            raise ValueError("token logprob is invalid")


@dataclass(frozen=True, slots=True)
class TeacherForcedObservation:
    case_id: str
    input_ids_sha256: str
    score_start_index: int
    target_kind: TeacherForcedTargetKind
    token_logprobs: tuple[TokenLogprob, ...]
    negative_log_likelihood: float
    mean_negative_log_likelihood: float
    perplexity: float
    response_sha256: str

    def __post_init__(self) -> None:
        _validate_case_id(self.case_id)
        _validate_sha256(self.input_ids_sha256, "input token IDs")
        _validate_sha256(self.response_sha256, "response")
        if (
            not self.token_logprobs
            or self.score_start_index <= 0
            or not math.isfinite(self.negative_log_likelihood)
            or not math.isfinite(self.mean_negative_log_likelihood)
            or not math.isfinite(self.perplexity)
            or self.perplexity <= 0.0
        ):
            raise ValueError("teacher-forced observation is invalid")


@dataclass(frozen=True, slots=True)
class TopKProbeObservation:
    case_id: str
    input_ids_sha256: str
    generated_token: TokenLogprob
    top_logprobs: tuple[TokenLogprob, ...]
    response_sha256: str

    def __post_init__(self) -> None:
        _validate_case_id(self.case_id)
        _validate_sha256(self.input_ids_sha256, "input token IDs")
        _validate_sha256(self.response_sha256, "response")
        if not self.top_logprobs or len(
            {entry.token_id for entry in self.top_logprobs}
        ) != len(self.top_logprobs):
            raise ValueError("top-k observation is invalid")


@dataclass(frozen=True, slots=True)
class SpeculativeGenerationEvidence:
    """Final native-response evidence for one speculative generation."""

    completion_token_count: int
    accepted_draft_token_count: int
    draft_token_count: int
    rejected_draft_token_count: int
    verification_pass_count: int
    acceptance_rate: float
    average_tokens_per_verification: float
    acceptance_histogram: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if (
            self.completion_token_count <= 0
            or self.accepted_draft_token_count < 0
            or self.draft_token_count <= 0
            or self.rejected_draft_token_count < 0
            or self.verification_pass_count <= 0
            or self.accepted_draft_token_count > self.draft_token_count
            or self.rejected_draft_token_count
            != self.draft_token_count - self.accepted_draft_token_count
            or not math.isfinite(self.acceptance_rate)
            or not 0.0 <= self.acceptance_rate <= 1.0
            or not math.isfinite(self.average_tokens_per_verification)
            or self.average_tokens_per_verification <= 0.0
            or (
                self.acceptance_histogram is not None
                and (
                    not self.acceptance_histogram
                    or any(value < 0 for value in self.acceptance_histogram)
                )
            )
        ):
            raise ValueError("speculative generation evidence is invalid")


@dataclass(frozen=True, slots=True)
class GenerationObservation:
    case_id: str
    input_ids_sha256: str
    output_ids: tuple[int, ...]
    text: str
    response_sha256: str
    speculative_decoding: SpeculativeGenerationEvidence | None

    def __post_init__(self) -> None:
        _validate_case_id(self.case_id)
        _validate_sha256(self.input_ids_sha256, "input token IDs")
        _validate_sha256(self.response_sha256, "response")
        _validate_token_ids(self.output_ids, minimum_count=1)
        if (
            self.speculative_decoding is not None
            and self.speculative_decoding.completion_token_count != len(self.output_ids)
        ):
            raise ValueError(
                "speculative evidence does not match generation token count"
            )


def parse_teacher_forced_response(
    request: TeacherForcedRequest,
    payload: JsonObject,
) -> TeacherForcedObservation:
    parsed, response_sha256 = _validated_response(
        payload, _TeacherForcedResponse, "teacher-forced response"
    )
    entries = parsed.meta_info.input_token_logprobs
    output_entries = parsed.meta_info.output_token_logprobs
    wire_score_start_index = request.score_start_index - 1
    if (
        parsed.meta_info.prompt_tokens != len(request.input_ids)
        or parsed.meta_info.finish_reason.length != 1
        or len(parsed.output_ids) != 1
        or len(entries) != len(request.input_ids) - wire_score_start_index
        or tuple(entry[1] for entry in entries)
        != request.input_ids[wire_score_start_index:]
        or len(output_entries) != 1
        or output_entries[0][1] != parsed.output_ids[0]
    ):
        raise Glm52MtpQualityError(
            "teacher-forced response does not match its exact request"
        )
    _validate_input_logprob_sentinel(entries[0])
    finite_entries = entries[1:]
    for entry in (*finite_entries, *output_entries):
        _validate_serialized_logprob(entry)
    token_logprobs = tuple(
        TokenLogprob(token_id=token_id, logprob=logprob)
        for logprob, token_id, _text in finite_entries
        if logprob is not None
    )
    if (
        len(token_logprobs) != len(request.input_ids) - request.score_start_index
        or tuple(entry.token_id for entry in token_logprobs)
        != request.input_ids[request.score_start_index :]
    ):
        raise Glm52MtpQualityError(
            "teacher-forced response has incomplete finite target logprobs"
        )
    negative_log_likelihood = -math.fsum(entry.logprob for entry in token_logprobs)
    mean_negative_log_likelihood = negative_log_likelihood / len(token_logprobs)
    try:
        perplexity = math.exp(mean_negative_log_likelihood)
    except OverflowError as error:
        raise Glm52MtpQualityError("teacher-forced perplexity is not finite") from error
    if not math.isfinite(perplexity):
        raise Glm52MtpQualityError("teacher-forced perplexity is not finite")
    return TeacherForcedObservation(
        case_id=request.case_id,
        input_ids_sha256=token_ids_sha256(request.input_ids),
        score_start_index=request.score_start_index,
        target_kind=request.target_kind,
        token_logprobs=token_logprobs,
        negative_log_likelihood=negative_log_likelihood,
        mean_negative_log_likelihood=mean_negative_log_likelihood,
        perplexity=perplexity,
        response_sha256=response_sha256,
    )


def parse_top_k_probe_response(
    request: TopKProbeRequest,
    payload: JsonObject,
) -> TopKProbeObservation:
    parsed, response_sha256 = _validated_response(
        payload, _TopKProbeResponse, "top-k probe response"
    )
    generated_entries = parsed.meta_info.output_token_logprobs
    top_groups = parsed.meta_info.output_top_logprobs
    if (
        parsed.meta_info.prompt_tokens != len(request.input_ids)
        or parsed.meta_info.finish_reason.length != 1
        or len(parsed.output_ids) != 1
        or len(generated_entries) != 1
        or generated_entries[0][1] != parsed.output_ids[0]
        or len(top_groups) != 1
        or len(top_groups[0]) != request.top_k
    ):
        raise Glm52MtpQualityError("top-k response does not match its exact request")
    generated_entry = generated_entries[0]
    top_entries = top_groups[0]
    for entry in (generated_entry, *top_entries):
        _validate_serialized_logprob(entry)
    if (
        len({entry[1] for entry in top_entries}) != len(top_entries)
        or any(
            current[0] < following[0]
            for current, following in zip(top_entries, top_entries[1:], strict=False)
        )
        or generated_entry not in top_entries
        or generated_entry[0] != top_entries[0][0]
    ):
        raise Glm52MtpQualityError("top-k response is incomplete or not ranked")
    return TopKProbeObservation(
        case_id=request.case_id,
        input_ids_sha256=token_ids_sha256(request.input_ids),
        generated_token=TokenLogprob(
            token_id=generated_entry[1],
            logprob=generated_entry[0],
        ),
        top_logprobs=tuple(
            TokenLogprob(token_id=token_id, logprob=logprob)
            for logprob, token_id, _text in top_entries
        ),
        response_sha256=response_sha256,
    )


def _parse_speculative_generation_evidence(
    payload: JsonObject,
    *,
    completion_token_count: int,
) -> SpeculativeGenerationEvidence | None:
    raw_meta_info = payload.get("meta_info")
    if not isinstance(raw_meta_info, dict):
        raise Glm52MtpQualityError("deterministic generation response lacks meta_info")

    # Keep one strict interpretation of SGLang's speculative response fields
    # across the quality gate and the throughput benchmark.
    from scripts import run_sglang_kt_glm52_tp2_local_benchmark as benchmark

    try:
        parsed = benchmark.parse_speculative_decoding_metrics(raw_meta_info)
    except benchmark.Glm52Tp2BenchmarkError as error:
        raise Glm52MtpQualityError(
            "deterministic generation has invalid speculative metrics"
        ) from error
    if parsed is None:
        return None

    required_counts = (
        parsed.spec_accept_token_num,
        parsed.spec_draft_token_num,
        parsed.spec_verify_ct,
    )
    if (
        parsed.spec_accept_rate is None
        or parsed.spec_accept_length is None
        or any(value is None for value in required_counts)
    ):
        raise Glm52MtpQualityError(
            "deterministic generation has incomplete speculative metrics"
        )
    accepted_token_count = cast(int, parsed.spec_accept_token_num)
    draft_token_count = cast(int, parsed.spec_draft_token_num)
    verification_pass_count = cast(int, parsed.spec_verify_ct)
    if (
        draft_token_count <= 0
        or verification_pass_count <= 0
        or accepted_token_count > draft_token_count
        or completion_token_count < verification_pass_count
        or not math.isclose(
            parsed.spec_accept_rate,
            accepted_token_count / draft_token_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            parsed.spec_accept_length,
            completion_token_count / verification_pass_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise Glm52MtpQualityError(
            "deterministic generation speculative metrics are inconsistent"
        )

    histogram = parsed.spec_accept_histogram
    if histogram is not None and (
        sum(histogram) != verification_pass_count
        or sum(
            accepted_count * step_count
            for accepted_count, step_count in enumerate(histogram)
        )
        != accepted_token_count
    ):
        raise Glm52MtpQualityError(
            "deterministic generation speculative histogram is inconsistent"
        )
    return SpeculativeGenerationEvidence(
        completion_token_count=completion_token_count,
        accepted_draft_token_count=accepted_token_count,
        draft_token_count=draft_token_count,
        rejected_draft_token_count=draft_token_count - accepted_token_count,
        verification_pass_count=verification_pass_count,
        acceptance_rate=parsed.spec_accept_rate,
        average_tokens_per_verification=parsed.spec_accept_length,
        acceptance_histogram=histogram,
    )


def parse_generation_response(
    request: DeterministicGenerationRequest,
    payload: JsonObject,
) -> GenerationObservation:
    parsed, response_sha256 = _validated_response(
        payload, _GenerationResponse, "deterministic generation response"
    )
    if (
        parsed.meta_info.prompt_tokens != len(request.input_ids)
        or parsed.meta_info.completion_tokens != request.max_new_tokens
        or parsed.meta_info.finish_reason.length != request.max_new_tokens
        or len(parsed.output_ids) != request.max_new_tokens
    ):
        raise Glm52MtpQualityError(
            "deterministic generation response does not match its exact request"
        )
    _validate_token_ids(parsed.output_ids, minimum_count=request.max_new_tokens)
    return GenerationObservation(
        case_id=request.case_id,
        input_ids_sha256=token_ids_sha256(request.input_ids),
        output_ids=parsed.output_ids,
        text=parsed.text,
        response_sha256=response_sha256,
        speculative_decoding=_parse_speculative_generation_evidence(
            payload,
            completion_token_count=parsed.meta_info.completion_tokens,
        ),
    )


@dataclass(frozen=True, slots=True)
class MarlinRuntimeControls:
    """Observed launch controls; implicit ``auto`` is never accepted."""

    configured_backend: Literal["marlin"]
    environment_backend: Literal["marlin"]
    return_original_logprob_environment: Literal["1"]

    def __post_init__(self) -> None:
        if (
            self.configured_backend != "marlin"
            or self.environment_backend != "marlin"
            or self.return_original_logprob_environment != "1"
        ):
            raise ValueError(
                "quality runs require explicit Marlin and original target logprobs"
            )


@dataclass(frozen=True, slots=True)
class MtpConfiguration:
    enabled: bool
    speculative_algorithm: Literal["EAGLE"] | None
    speculative_num_steps: int | None
    speculative_eagle_topk: int | None
    speculative_num_draft_tokens: int | None

    def __post_init__(self) -> None:
        values = (
            self.speculative_num_steps,
            self.speculative_eagle_topk,
            self.speculative_num_draft_tokens,
        )
        if self.enabled:
            if self.speculative_algorithm != "EAGLE" or any(
                value is None or value <= 0 for value in values
            ):
                raise ValueError("enabled MTP configuration is incomplete")
        elif self.speculative_algorithm is not None or any(
            value is not None for value in values
        ):
            raise ValueError("disabled MTP configuration has speculative fields")


def _validate_nested_backend_controls(value: JsonValue) -> None:
    if isinstance(value, list):
        for item in value:
            _validate_nested_backend_controls(item)
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        normalized_key = key.casefold().replace("-", "_")
        if (
            normalized_key
            in {
                "mla_kv_b_w8_backend",
                "sglang_mla_kv_b_w8_backend",
                "requested_backend",
                "expected_runtime_backend",
            }
            and item != "marlin"
        ):
            raise ValueError("historical or implicit compact MLA backend is forbidden")
        if normalized_key == "sglang_return_original_logprob" and item not in {"1", 1}:
            raise ValueError("original target logprob control is not enabled")
        _validate_nested_backend_controls(item)


@dataclass(frozen=True, slots=True)
class RunContract:
    """All pair-invariant launch data plus the sole allowed MTP difference."""

    common: JsonObject
    artifact: JsonObject
    runtime_controls: MarlinRuntimeControls
    mtp: MtpConfiguration

    def __post_init__(self) -> None:
        if not self.common or not self.artifact:
            raise ValueError("run contract and artifact identity must be nonempty")
        common_copy = json.loads(_canonical_json_bytes(self.common))
        artifact_copy = json.loads(_canonical_json_bytes(self.artifact))
        if not isinstance(common_copy, dict) or not isinstance(artifact_copy, dict):
            raise ValueError("run contract sections must be JSON objects")
        typed_common = cast(JsonObject, common_copy)
        typed_artifact = cast(JsonObject, artifact_copy)
        _validate_nested_backend_controls(typed_common)
        _validate_nested_backend_controls(typed_artifact)
        object.__setattr__(self, "common", typed_common)
        object.__setattr__(self, "artifact", typed_artifact)

    @property
    def common_sha256(self) -> str:
        return _canonical_sha256(self.common)

    @property
    def artifact_sha256(self) -> str:
        return _canonical_sha256(self.artifact)


@dataclass(frozen=True, slots=True)
class MarlinRankObservation:
    tensor_parallel_rank: Literal[0, 1]
    backend: Literal["marlin"]
    module_count: int
    local_heads_per_module: int

    def __post_init__(self) -> None:
        if (
            self.tensor_parallel_rank not in {0, 1}
            or self.backend != "marlin"
            or self.module_count <= 0
            or self.local_heads_per_module != MARLIN_LOCAL_HEADS_PER_MODULE
        ):
            raise ValueError("Marlin rank observation is invalid")


@dataclass(frozen=True, slots=True)
class MarlinCensus:
    mtp_enabled: bool
    requested_backend: Literal["marlin"]
    expected_runtime_backend: Literal["marlin"]
    passed: Literal[True]
    observations: tuple[MarlinRankObservation, MarlinRankObservation]

    def __post_init__(self) -> None:
        expected_modules = TARGET_LAYER_COUNT + (
            MTP_LAYER_COUNT if self.mtp_enabled else 0
        )
        if (
            self.requested_backend != "marlin"
            or self.expected_runtime_backend != "marlin"
            or self.passed is not True
            or tuple(item.tensor_parallel_rank for item in self.observations) != (0, 1)
            or any(
                item.backend != "marlin"
                or item.module_count != expected_modules
                or item.local_heads_per_module != MARLIN_LOCAL_HEADS_PER_MODULE
                for item in self.observations
            )
        ):
            raise ValueError("Marlin census does not prove the exact TP2 layer count")


@dataclass(frozen=True, slots=True)
class SpeculativeDecodingExerciseEvidence:
    """Aggregate proof that native generation did or did not exercise MTP."""

    source: Literal["native_non_stream_generate_meta_info"]
    mtp_enabled: bool
    generation_request_count: int
    generation_requests_with_metrics: int
    completion_token_count: int
    accepted_draft_token_count: int
    draft_token_count: int
    rejected_draft_token_count: int
    verification_pass_count: int
    acceptance_rate: float | None
    average_tokens_per_verification: float | None

    def __post_init__(self) -> None:
        common_invalid = (
            self.source != "native_non_stream_generate_meta_info"
            or self.generation_request_count <= 0
            or not 0
            <= self.generation_requests_with_metrics
            <= self.generation_request_count
            or self.completion_token_count <= 0
            or self.accepted_draft_token_count < 0
            or self.draft_token_count < 0
            or self.rejected_draft_token_count < 0
            or self.verification_pass_count < 0
            or self.draft_token_count
            != self.accepted_draft_token_count + self.rejected_draft_token_count
        )
        enabled_invalid = self.mtp_enabled and (
            self.generation_requests_with_metrics != self.generation_request_count
            or self.draft_token_count <= 0
            or self.verification_pass_count <= 0
            or self.acceptance_rate is None
            or not math.isfinite(self.acceptance_rate)
            or not 0.0 <= self.acceptance_rate <= 1.0
            or self.average_tokens_per_verification is None
            or not math.isfinite(self.average_tokens_per_verification)
            or self.average_tokens_per_verification <= 0.0
        )
        disabled_invalid = not self.mtp_enabled and (
            self.generation_requests_with_metrics != 0
            or self.accepted_draft_token_count != 0
            or self.draft_token_count != 0
            or self.rejected_draft_token_count != 0
            or self.verification_pass_count != 0
            or self.acceptance_rate is not None
            or self.average_tokens_per_verification is not None
        )
        if common_invalid or enabled_invalid or disabled_invalid:
            raise ValueError("speculative decoding exercise evidence is invalid")


def _aggregate_speculative_decoding_evidence(
    mtp: MtpConfiguration,
    generations: tuple[GenerationObservation, ...],
) -> SpeculativeDecodingExerciseEvidence:
    observations = tuple(
        observation.speculative_decoding
        for observation in generations
        if observation.speculative_decoding is not None
    )
    completion_token_count = sum(
        len(observation.output_ids) for observation in generations
    )
    if not mtp.enabled:
        if observations:
            raise ValueError("MTP-off run unexpectedly returned speculative metrics")
        return SpeculativeDecodingExerciseEvidence(
            source="native_non_stream_generate_meta_info",
            mtp_enabled=False,
            generation_request_count=len(generations),
            generation_requests_with_metrics=0,
            completion_token_count=completion_token_count,
            accepted_draft_token_count=0,
            draft_token_count=0,
            rejected_draft_token_count=0,
            verification_pass_count=0,
            acceptance_rate=None,
            average_tokens_per_verification=None,
        )

    if len(observations) != len(generations):
        raise ValueError(
            "MTP-on run lacks speculative metrics for every generation request"
        )
    draft_tokens_per_verification = cast(int, mtp.speculative_num_draft_tokens) - 1
    if draft_tokens_per_verification <= 0 or any(
        observation.draft_token_count
        != observation.verification_pass_count * draft_tokens_per_verification
        for observation in observations
    ):
        raise ValueError(
            "MTP-on speculative draft counts do not match the launch contract"
        )

    accepted_draft_token_count = sum(
        observation.accepted_draft_token_count for observation in observations
    )
    draft_token_count = sum(
        observation.draft_token_count for observation in observations
    )
    rejected_draft_token_count = sum(
        observation.rejected_draft_token_count for observation in observations
    )
    verification_pass_count = sum(
        observation.verification_pass_count for observation in observations
    )
    return SpeculativeDecodingExerciseEvidence(
        source="native_non_stream_generate_meta_info",
        mtp_enabled=True,
        generation_request_count=len(generations),
        generation_requests_with_metrics=len(observations),
        completion_token_count=completion_token_count,
        accepted_draft_token_count=accepted_draft_token_count,
        draft_token_count=draft_token_count,
        rejected_draft_token_count=rejected_draft_token_count,
        verification_pass_count=verification_pass_count,
        acceptance_rate=accepted_draft_token_count / draft_token_count,
        average_tokens_per_verification=(
            completion_token_count / verification_pass_count
        ),
    )


@dataclass(frozen=True, slots=True)
class QualityRun:
    run_id: str
    profile: QualityProfile
    contract: RunContract
    backend_census: MarlinCensus
    teacher_forced: tuple[TeacherForcedObservation, ...]
    top_k_probes: tuple[TopKProbeObservation, ...]
    generations: tuple[GenerationObservation, ...]
    speculative_decoding_evidence: SpeculativeDecodingExerciseEvidence = field(
        init=False
    )

    def __post_init__(self) -> None:
        if (
            not self.run_id
            or self.contract.mtp.enabled != self.backend_census.mtp_enabled
        ):
            raise ValueError("quality run identity is invalid")
        expected_case_ids = self.profile.case_ids
        groups: tuple[
            tuple[
                TeacherForcedObservation | TopKProbeObservation | GenerationObservation,
                ...,
            ],
            ...,
        ] = (self.teacher_forced, self.top_k_probes, self.generations)
        if any(
            tuple(item.case_id for item in group) != expected_case_ids
            for group in groups
        ):
            raise ValueError("quality run observations do not match the profile")
        for probe, generation in zip(self.top_k_probes, self.generations, strict=True):
            if probe.input_ids_sha256 != generation.input_ids_sha256:
                raise ValueError("probe and generation case inputs are not identical")
        object.__setattr__(
            self,
            "speculative_decoding_evidence",
            _aggregate_speculative_decoding_evidence(
                self.contract.mtp,
                self.generations,
            ),
        )


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    maximum_teacher_token_logprob_absolute_delta: float = 0.005
    maximum_probe_logprob_absolute_delta: float = 0.005
    maximum_mean_negative_log_likelihood_absolute_delta: float = 0.001
    maximum_perplexity_relative_delta: float = 0.001

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.maximum_teacher_token_logprob_absolute_delta,
                self.maximum_probe_logprob_absolute_delta,
                self.maximum_mean_negative_log_likelihood_absolute_delta,
                self.maximum_perplexity_relative_delta,
            )
        ):
            raise ValueError("quality thresholds must be finite and nonnegative")


DEFAULT_QUALITY_THRESHOLDS: Final = QualityThresholds()


@dataclass(frozen=True, slots=True)
class MatchedQualityMetrics:
    case_count: int
    prompt_self_likelihood_case_count: int
    reference_completion_case_count: int
    teacher_forced_token_count: int
    generation_token_count: int
    mtp_off_negative_log_likelihood: float
    mtp_on_negative_log_likelihood: float
    negative_log_likelihood_delta: float
    mtp_off_mean_negative_log_likelihood: float
    mtp_on_mean_negative_log_likelihood: float
    mean_negative_log_likelihood_delta: float
    mtp_off_perplexity: float
    mtp_on_perplexity: float
    perplexity_delta: float
    perplexity_relative_delta: float
    maximum_teacher_token_logprob_absolute_delta: float
    maximum_probe_logprob_absolute_delta: float


@dataclass(frozen=True, slots=True)
class MatchedQualityComparison:
    status: Literal["passed"]
    mtp_off_run_id: str
    mtp_on_run_id: str
    common_contract_sha256: str
    artifact_contract_sha256: str
    quality_profile_sha256: str
    metrics: MatchedQualityMetrics
    thresholds: QualityThresholds

    def json_object(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


def _require_equal_pair_contracts(
    mtp_off: QualityRun,
    mtp_on: QualityRun,
) -> None:
    if mtp_off.contract.mtp.enabled or not mtp_on.contract.mtp.enabled:
        raise Glm52MtpQualityError("matched pair must be ordered MTP-off then MTP-on")
    if (
        mtp_off.profile != mtp_on.profile
        or mtp_off.contract.common_sha256 != mtp_on.contract.common_sha256
        or mtp_off.contract.artifact_sha256 != mtp_on.contract.artifact_sha256
        or mtp_off.contract.runtime_controls != mtp_on.contract.runtime_controls
    ):
        raise Glm52MtpQualityError(
            "matched pair differs outside the explicit MTP configuration"
        )
    if (
        mtp_off.contract.runtime_controls.configured_backend != "marlin"
        or mtp_off.contract.runtime_controls.environment_backend != "marlin"
        or mtp_off.contract.runtime_controls.return_original_logprob_environment != "1"
    ):
        raise Glm52MtpQualityError(
            "matched pair lacks explicit Marlin logprob controls"
        )


def compare_matched_runs(
    mtp_off: QualityRun,
    mtp_on: QualityRun,
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
) -> MatchedQualityComparison:
    """Require exact greedy parity and bounded target-logprob drift."""

    _require_equal_pair_contracts(mtp_off, mtp_on)
    teacher_logprob_deltas: list[float] = []
    probe_logprob_deltas: list[float] = []
    off_negative_log_likelihood = 0.0
    on_negative_log_likelihood = 0.0
    teacher_token_count = 0
    generation_token_count = 0

    for off_observation, on_observation in zip(
        mtp_off.teacher_forced,
        mtp_on.teacher_forced,
        strict=True,
    ):
        if (
            off_observation.case_id != on_observation.case_id
            or off_observation.input_ids_sha256 != on_observation.input_ids_sha256
            or off_observation.score_start_index != on_observation.score_start_index
            or off_observation.target_kind != on_observation.target_kind
            or tuple(item.token_id for item in off_observation.token_logprobs)
            != tuple(item.token_id for item in on_observation.token_logprobs)
        ):
            raise Glm52MtpQualityError("teacher-forced evidence is not token-aligned")
        teacher_logprob_deltas.extend(
            abs(off_entry.logprob - on_entry.logprob)
            for off_entry, on_entry in zip(
                off_observation.token_logprobs,
                on_observation.token_logprobs,
                strict=True,
            )
        )
        teacher_token_count += len(off_observation.token_logprobs)
        off_negative_log_likelihood += off_observation.negative_log_likelihood
        on_negative_log_likelihood += on_observation.negative_log_likelihood

    for off_observation, on_observation in zip(
        mtp_off.top_k_probes,
        mtp_on.top_k_probes,
        strict=True,
    ):
        if (
            off_observation.case_id != on_observation.case_id
            or off_observation.input_ids_sha256 != on_observation.input_ids_sha256
            or off_observation.generated_token.token_id
            != on_observation.generated_token.token_id
            or tuple(item.token_id for item in off_observation.top_logprobs)
            != tuple(item.token_id for item in on_observation.top_logprobs)
        ):
            raise Glm52MtpQualityError(
                "top-k evidence lacks exact greedy/token-rank parity"
            )
        probe_logprob_deltas.append(
            abs(
                off_observation.generated_token.logprob
                - on_observation.generated_token.logprob
            )
        )
        probe_logprob_deltas.extend(
            abs(off_entry.logprob - on_entry.logprob)
            for off_entry, on_entry in zip(
                off_observation.top_logprobs,
                on_observation.top_logprobs,
                strict=True,
            )
        )

    for off_observation, on_observation in zip(
        mtp_off.generations,
        mtp_on.generations,
        strict=True,
    ):
        if (
            off_observation.case_id != on_observation.case_id
            or off_observation.input_ids_sha256 != on_observation.input_ids_sha256
            or off_observation.output_ids != on_observation.output_ids
        ):
            raise Glm52MtpQualityError("deterministic greedy token parity failed")
        generation_token_count += len(off_observation.output_ids)

    if teacher_token_count <= 0 or not teacher_logprob_deltas:
        raise Glm52MtpQualityError("matched pair has no teacher-forced evidence")
    off_mean_negative_log_likelihood = off_negative_log_likelihood / teacher_token_count
    on_mean_negative_log_likelihood = on_negative_log_likelihood / teacher_token_count
    try:
        off_perplexity = math.exp(off_mean_negative_log_likelihood)
        on_perplexity = math.exp(on_mean_negative_log_likelihood)
    except OverflowError as error:
        raise Glm52MtpQualityError("matched perplexity is not finite") from error
    if not math.isfinite(off_perplexity) or not math.isfinite(on_perplexity):
        raise Glm52MtpQualityError("matched perplexity is not finite")

    maximum_teacher_delta = max(teacher_logprob_deltas)
    maximum_probe_delta = max(probe_logprob_deltas, default=0.0)
    mean_negative_log_likelihood_delta = (
        on_mean_negative_log_likelihood - off_mean_negative_log_likelihood
    )
    perplexity_delta = on_perplexity - off_perplexity
    perplexity_relative_delta = perplexity_delta / off_perplexity
    if (
        maximum_teacher_delta > thresholds.maximum_teacher_token_logprob_absolute_delta
        or maximum_probe_delta > thresholds.maximum_probe_logprob_absolute_delta
        or abs(mean_negative_log_likelihood_delta)
        > thresholds.maximum_mean_negative_log_likelihood_absolute_delta
        or abs(perplexity_relative_delta) > thresholds.maximum_perplexity_relative_delta
    ):
        raise Glm52MtpQualityError(
            "matched target-logprob quality thresholds were exceeded"
        )

    return MatchedQualityComparison(
        status="passed",
        mtp_off_run_id=mtp_off.run_id,
        mtp_on_run_id=mtp_on.run_id,
        common_contract_sha256=mtp_off.contract.common_sha256,
        artifact_contract_sha256=mtp_off.contract.artifact_sha256,
        quality_profile_sha256=mtp_off.profile.content_sha256,
        metrics=MatchedQualityMetrics(
            case_count=len(mtp_off.profile.case_ids),
            prompt_self_likelihood_case_count=sum(
                observation.target_kind == "prompt_self_likelihood"
                for observation in mtp_off.teacher_forced
            ),
            reference_completion_case_count=sum(
                observation.target_kind == "reference_completion"
                for observation in mtp_off.teacher_forced
            ),
            teacher_forced_token_count=teacher_token_count,
            generation_token_count=generation_token_count,
            mtp_off_negative_log_likelihood=off_negative_log_likelihood,
            mtp_on_negative_log_likelihood=on_negative_log_likelihood,
            negative_log_likelihood_delta=(
                on_negative_log_likelihood - off_negative_log_likelihood
            ),
            mtp_off_mean_negative_log_likelihood=(off_mean_negative_log_likelihood),
            mtp_on_mean_negative_log_likelihood=on_mean_negative_log_likelihood,
            mean_negative_log_likelihood_delta=(mean_negative_log_likelihood_delta),
            mtp_off_perplexity=off_perplexity,
            mtp_on_perplexity=on_perplexity,
            perplexity_delta=perplexity_delta,
            perplexity_relative_delta=perplexity_relative_delta,
            maximum_teacher_token_logprob_absolute_delta=maximum_teacher_delta,
            maximum_probe_logprob_absolute_delta=maximum_probe_delta,
        ),
        thresholds=thresholds,
    )


def capture_quality_run(
    generate: NativeGenerate,
    cases: Sequence[TokenizedQualityCase],
    *,
    run_id: str,
    contract: RunContract,
    backend_census: MarlinCensus,
    profile: QualityProfile = THIN_FIRST_RUN_PROFILE,
) -> QualityRun:
    """Capture the three bounded native observations for every profile case."""

    case_by_id: dict[str, TokenizedQualityCase] = {}
    for item in cases:
        if item.case_id in case_by_id:
            raise Glm52MtpQualityError(f"duplicate quality case {item.case_id}")
        case_by_id[item.case_id] = item
    missing_case_ids = [
        case_id for case_id in profile.case_ids if case_id not in case_by_id
    ]
    if missing_case_ids:
        raise Glm52MtpQualityError(
            f"quality profile is missing cases: {missing_case_ids}"
        )

    teacher_forced: list[TeacherForcedObservation] = []
    top_k_probes: list[TopKProbeObservation] = []
    generations: list[GenerationObservation] = []
    for case_id in profile.case_ids:
        quality_case = case_by_id[case_id]
        teacher_request = TeacherForcedRequest(
            case_id=case_id,
            input_ids=cast(
                tuple[int, ...],
                quality_case.teacher_forced_input_ids,
            ),
            score_start_index=quality_case.teacher_forced_score_start_index,
            target_kind=quality_case.teacher_forced_target_kind,
            sampling_seed=profile.sampling_seed,
        )
        teacher_forced.append(
            parse_teacher_forced_response(
                teacher_request,
                generate(teacher_request.json_object()),
            )
        )
        probe_request = TopKProbeRequest(
            case_id=case_id,
            input_ids=quality_case.input_ids,
            top_k=profile.top_k,
            sampling_seed=profile.sampling_seed,
        )
        top_k_probes.append(
            parse_top_k_probe_response(
                probe_request,
                generate(probe_request.json_object()),
            )
        )
        generation_request = DeterministicGenerationRequest(
            case_id=case_id,
            input_ids=quality_case.input_ids,
            max_new_tokens=profile.generation_max_new_tokens,
            sampling_seed=profile.sampling_seed,
        )
        generations.append(
            parse_generation_response(
                generation_request,
                generate(generation_request.json_object()),
            )
        )

    return QualityRun(
        run_id=run_id,
        profile=profile,
        contract=contract,
        backend_census=backend_census,
        teacher_forced=tuple(teacher_forced),
        top_k_probes=tuple(top_k_probes),
        generations=tuple(generations),
    )
