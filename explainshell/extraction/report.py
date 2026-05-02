"""Structured report for extraction runs.

Written to the run directory as ``report.json`` after extraction completes.
"""

from __future__ import annotations

import statistics
from typing import Any, Literal

from pydantic import BaseModel, Field


class GitInfo(BaseModel):
    commit: str | None
    commit_short: str | None
    dirty: bool | None


class ExtractConfig(BaseModel):
    mode: str | None = None
    model: str | None = None
    overwrite: bool = False
    filter_db: str | None = None
    drop: bool = False
    jobs: int = 1
    batch_size: int | None = None
    debug: bool = False
    small_only: bool = False
    large_only: bool = False


class ExtractSummary(BaseModel):
    succeeded: int
    skipped: int
    failed: int
    prefilter_skipped: int = 0
    symlinks_mapped: int = 0
    content_deduped: int = 0
    interrupted: bool = False
    fatal_error: str | None = None


class DbCounts(BaseModel):
    manpages: int
    mappings: int


class TokenUsage(BaseModel):
    """Aggregate LLM token usage across all successful extractions in a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    chunks: int = 0
    plain_text_chars: int = 0


class OptionCountSummary(BaseModel):
    """Distribution of per-page option counts across successful extractions."""

    n: int
    total: int
    mean: float
    median: float
    p90: float
    max: int
    # Stable bucket order. Values are file counts.
    buckets: dict[str, int]

    @classmethod
    def empty(cls) -> OptionCountSummary:
        """Zero-everything summary, used when the histogram is unavailable.

        Old reports written before option-count tracking existed migrate
        to this shape rather than ``null`` so analysis tools can rely on
        the field always being a populated struct.
        """
        return cls(
            n=0,
            total=0,
            mean=0.0,
            median=0.0,
            p90=0.0,
            max=0,
            buckets={"0": 0, "1-5": 0, "6-15": 0, "16-50": 0, "50+": 0},
        )

    @classmethod
    def from_counts(cls, counts: list[int]) -> OptionCountSummary:
        """Build a summary from a list of per-file option counts.

        Empty input yields :meth:`empty` rather than ``None`` — the field
        is always present in the report.
        """
        if not counts:
            return cls.empty()
        buckets = {"0": 0, "1-5": 0, "6-15": 0, "16-50": 0, "50+": 0}
        for c in counts:
            if c == 0:
                buckets["0"] += 1
            elif c <= 5:
                buckets["1-5"] += 1
            elif c <= 15:
                buckets["6-15"] += 1
            elif c <= 50:
                buckets["16-50"] += 1
            else:
                buckets["50+"] += 1
        # statistics.quantiles requires n >= 2 data points.
        if len(counts) >= 2:
            p90 = statistics.quantiles(counts, n=10)[8]
        else:
            p90 = float(counts[0])
        return cls(
            n=len(counts),
            total=sum(counts),
            mean=round(statistics.mean(counts), 2),
            median=float(statistics.median(counts)),
            p90=round(float(p90), 2),
            max=max(counts),
            buckets=buckets,
        )


class FailureEntry(BaseModel):
    """One file that failed extraction, with classification."""

    path: str
    reason_class: str | None = None
    message: str


class SkipEntry(BaseModel):
    """One file the extractor intentionally skipped, with classification."""

    path: str
    reason_class: str | None = None
    message: str


class ExtractionReport(BaseModel):
    version: Literal[1] = 1
    command: Literal["extract"] = "extract"
    timestamp: str
    git: GitInfo
    config: ExtractConfig
    elapsed_seconds: float
    summary: ExtractSummary
    db_before: DbCounts
    db_after: DbCounts
    usage: TokenUsage = Field(default_factory=TokenUsage)
    option_counts: OptionCountSummary = Field(default_factory=OptionCountSummary.empty)
    failures: list[FailureEntry] = Field(default_factory=list)
    skips: list[SkipEntry] = Field(default_factory=list)
    batch_manifest: dict[str, Any] | None = None
