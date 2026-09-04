"""Beaker repository optimization spec for AutomationBench skills.

The candidate is the ``skills/`` tree. The harness, rubric, and this spec stay
fixed. Each case is one public AutomationBench task keyed by ``task_name``.
Scoring reuses the benchmark's own assertion results from ``run_one``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beaker import (
    STANDARD_JSONL_CASE_SCHEMA,
    Case,
    CaseDataLoader,
    CaseResult,
    CaseScore,
    Check,
    DatasetRowContext,
    OptimizationContext,
    Spec,
    inference_target,
    objective_score,
    spec,
)

from automationbench_skills.data.tasks import Sample, load_samples
from automationbench_skills.runner import DEFAULT_MODEL, ModelSpec, run_one_async


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"

TASK_COMPLETED_FIELD = "task_completed_correctly"
PARTIAL_CREDIT_FIELD = "partial_credit"
# Matches AutomationBench's own rubric: partial_credit is the training reward;
# task_completed_correctly is the official 0/1 pass rate and stays diagnostic.
FIELD_WEIGHTS = {
    PARTIAL_CREDIT_FIELD: 1.0,
    TASK_COMPLETED_FIELD: 0.0,
}


@dataclass(frozen=True)
class _TaskRow:
    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    group_key: str = "default"


class _TaskDataLoader(CaseDataLoader[_TaskRow]):
    """Validate standard JSONL rows for AutomationBench task names."""

    dataset_schema = STANDARD_JSONL_CASE_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> _TaskRow:
        del context
        missing = [name for name in ("id", "input", "expected") if name not in raw]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")
        row_id = str(raw["id"]).strip()
        if not row_id:
            raise ValueError("id must be non-empty")
        input_payload = raw["input"]
        expected = raw["expected"]
        metadata = raw.get("metadata") or {}
        if not isinstance(input_payload, Mapping):
            raise TypeError("input must be a JSON object")
        if not isinstance(expected, Mapping):
            raise TypeError("expected must be a JSON object")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a JSON object")
        task_name = str(input_payload.get("task_name") or row_id).strip()
        if not task_name:
            raise ValueError("input.task_name must be non-empty")
        return _TaskRow(
            id=row_id,
            input={"task_name": task_name, **dict(input_payload)},
            expected=dict(expected),
            metadata=dict(metadata),
            group_key=str(raw.get("group_key") or metadata.get("domain") or "default"),
        )

    def iter_cases(self, row: _TaskRow, context: DatasetRowContext) -> Iterable[Case]:
        del context
        yield Case(
            input=row.input,
            case_id=row.id,
            ground_truth=row.expected,
            group_key=row.group_key,
            metadata=row.metadata,
        )


def _sample_by_name(task_name: str) -> Sample:
    by_name = {sample.task_name: sample for sample in load_samples()}
    try:
        return by_name[task_name]
    except KeyError as exc:
        raise KeyError(f"unknown AutomationBench task_name: {task_name!r}") from exc


def _gateway_model_spec(base_url: str, api_key: str, model: str) -> ModelSpec:
    os.environ["BEAKER_INFERENCE_API_KEY"] = api_key
    return ModelSpec(
        name=model,
        base_url=base_url,
        api_key_var="BEAKER_INFERENCE_API_KEY",
        api="chat_completions",
    )


def _gateway_credentials() -> tuple[str, str] | None:
    """Return hosted gateway credentials, or None outside a Beaker rollout."""
    base_url = (os.environ.get("BEAKER_INFERENCE_BASE_URL") or "").rstrip("/")
    api_key = os.environ.get("BEAKER_INFERENCE_API_KEY") or os.environ.get("BEAKER_RUN_TOKEN")
    if not base_url:
        writeback = (os.environ.get("BEAKER_WRITEBACK_BASE_URL") or "").rstrip("/")
        if writeback:
            base_url = f"{writeback}/v1/llm"
    if base_url and api_key:
        return base_url, api_key
    return None


def _model_for_runtime(runtime: Any) -> ModelSpec:
    """Route every hosted rollout through the Beaker gateway.

    A selected ``runtime.model`` uses ``inference_target``. A hosted run with
    no selected model still uses the gateway and this recipe's default model.
    Local smoke and ordinary application runs keep the production client.
    """
    selected = getattr(runtime, "model", None) or getattr(runtime, "canonical_model_id", None)
    if selected:
        target = inference_target(runtime)
        return _gateway_model_spec(target.base_url, target.api_key, target.model)

    credentials = _gateway_credentials()
    if credentials is not None:
        base_url, api_key = credentials
        return _gateway_model_spec(base_url, api_key, f"openai:{DEFAULT_MODEL}")

    return ModelSpec()


def _trace_provider(model_name: str) -> str:
    if ":" in model_name:
        return model_name.split(":", 1)[0].lower()
    if "/" in model_name:
        return model_name.split("/", 1)[0].lower()
    return "openai"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return str(value)


async def _run_case(*, case: Case, targets: None, runtime: Any) -> CaseResult:
    del targets
    task_name = str((case.input or {}).get("task_name") or case.case_id or "").strip()
    if not task_name:
        return CaseResult.failed("case input is missing task_name")
    try:
        sample = _sample_by_name(task_name)
    except KeyError as exc:
        return CaseResult.failed(str(exc))

    model = _model_for_runtime(runtime)
    skills_dir = SKILLS_DIR if SKILLS_DIR.is_dir() else None
    with runtime.trace.stage(
        "automationbench.run_one",
        inputs={"task_name": task_name, "model": model.name},
    ) as stage:
        with runtime.trace.model_call(
            operation="chat.completions",
            provider=_trace_provider(model.name),
            model=model.name,
            input_messages=sample.prompt,
        ) as model_span:
            result = await run_one_async(sample, model=model, skills_dir=skills_dir)
            model_span.output(
                {
                    TASK_COMPLETED_FIELD: result.task_completed_correctly,
                    PARTIAL_CREDIT_FIELD: result.partial_credit,
                    "error": result.error,
                }
            )
        stage.output(
            {
                TASK_COMPLETED_FIELD: result.task_completed_correctly,
                PARTIAL_CREDIT_FIELD: result.partial_credit,
                "error": result.error,
            }
        )

    if result.error and not result.assertion_results and not result.trajectory:
        return CaseResult.failed(
            str(result.error),
            context={"error": str(result.error), "task_name": task_name},
            retryable="timeout" in str(result.error).lower(),
        )

    return CaseResult(
        output=None,
        output_kind="none",
        context={
            "task_name": result.task_name,
            "domain": result.domain,
            TASK_COMPLETED_FIELD: float(result.task_completed_correctly),
            PARTIAL_CREDIT_FIELD: float(result.partial_credit),
            "assertion_results": _jsonable(result.assertion_results),
            "trajectory": _jsonable(result.trajectory),
            "end_state": _jsonable(result.end_state),
            "error": None if result.error is None else str(result.error),
        },
    )


class _AssertionScorer:
    """Score one rollout from the benchmark's assertion results and metrics."""

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        context = result.context if isinstance(result.context, Mapping) else {}
        outcomes = context.get("assertion_results") or []
        if not isinstance(outcomes, list):
            outcomes = []

        checks: list[Check] = []
        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                continue
            assertion_type = str(outcome.get("type") or "assertion")
            params = outcome.get("params") if isinstance(outcome.get("params"), Mapping) else {}
            excluded = bool(outcome.get("excluded"))
            passed = bool(outcome.get("passed"))
            checks.append(
                Check(
                    name=assertion_type,
                    description=str(params) if params else None,
                    verdict="pass" if passed else "fail",
                    informational=excluded,
                    group=str(context.get("domain") or case.group_key or ""),
                )
            )

        expected_assertions = (case.ground_truth or {}).get("assertions") or []
        if not checks and isinstance(expected_assertions, list):
            for assertion in expected_assertions:
                if not isinstance(assertion, Mapping):
                    continue
                checks.append(
                    Check(
                        name=str(assertion.get("type") or "assertion"),
                        description="assertion was not evaluated; rollout produced no results",
                        verdict="fail",
                        informational=bool(assertion.get("scored") is False or assertion.get("excluded") is True),
                    )
                )

        partial = context.get(PARTIAL_CREDIT_FIELD)
        completed = context.get(TASK_COMPLETED_FIELD)
        if not isinstance(partial, (int, float)):
            scored = [check for check in checks if not check.informational]
            partial = (sum(1.0 for check in scored if check.verdict == "pass") / len(scored)) if scored else 0.0
        if not isinstance(completed, (int, float)):
            completed = 1.0 if float(partial) == 1.0 else 0.0

        field_scores = {
            PARTIAL_CREDIT_FIELD: float(partial),
            TASK_COMPLETED_FIELD: float(completed),
        }
        checks.append(
            Check(
                name=PARTIAL_CREDIT_FIELD,
                verdict=float(partial),
                message="fraction of scored assertions that passed",
            )
        )
        checks.append(
            Check(
                name=TASK_COMPLETED_FIELD,
                verdict="pass" if float(completed) == 1.0 else "fail",
                message="official 0/1 benchmark pass",
                informational=True,
            )
        )
        return CaseScore(
            field_scores=field_scores,
            objective=objective_score(field_scores, field_weights=FIELD_WEIGHTS),
            checks=tuple(checks),
        )


@spec(
    dataset_schema=STANDARD_JSONL_CASE_SCHEMA,
    repository=("skills",),
)
def build_spec(ctx: OptimizationContext) -> Spec:
    del ctx
    return Spec(
        data_loader=_TaskDataLoader(),
        run_case=_run_case,
        scorer=_AssertionScorer(),
    )
