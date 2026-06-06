"""Risk auditor: takes retrieval hits, returns a structured risk report.

The auditor is the "G" in RAG. We don't rely on a hosted LLM: instead, the
auditor uses a hand-curated library of legal-risk pattern families and
matches them against the retrieved chunks. This keeps the pipeline
fully offline, fully deterministic, and free of model hallucinations.

Each risk family is described by:
  * A human-readable name (shown in the report).
  * A severity tier (LOW / MEDIUM / HIGH / CRITICAL).
  * A list of regex patterns that flag candidate clauses.
  * A list of "good" patterns that, if present in the same chunk, demote
    the finding (so a chunk that mentions arbitration but also explicitly
    says the user retains the right to opt out is flagged lower than a
    chunk that says "binding arbitration, no exceptions").
  * A short description of why the family matters, used in the report.

The auditor runs once per query and returns a RiskAssessment containing
all triggered findings, ranked by severity.

Two-stage match (this is the fix for the "false positive family"
issue from the QA pass):
  1. PRIMARY match: run patterns against `chunk.text` - the sentence
     the retriever actually matched. If the user's question was about
     "content licensing" and the matched sentence is about content
     licensing, the patterns that fire will be the content-licensing
     family. This is the high-confidence path.
  2. CO-LOCATED match: run patterns against `chunk.parent_text` (the
     full paragraph) and only keep findings whose trigger text appears
     OUTSIDE the matched sentence. These get marked "co-located" and
     their severity is demoted by one tier. This is how we still tell
     the user "this paragraph also has a data-selling clause nearby"
     without confusing it with the question they actually asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from src.retrieval.retriever import RetrievedChunk
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


class RiskSeverity(str, Enum):
    """Ordered from least to most severe so we can sort findings."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}[self.value]


class MatchLocation(str, Enum):
    """Where in the chunk the pattern was found."""

    PRIMARY = "primary"      # trigger text is in the matched sentence
    CO_LOCATED = "co-located"  # trigger text is in a sibling sentence of the same paragraph

    def __str__(self) -> str:
        return self.value


@dataclass
class RiskPattern:
    """A single legal-risk pattern family."""

    family: str
    severity: RiskSeverity
    description: str
    trigger_patterns: tuple[str, ...]
    mitigating_patterns: tuple[str, ...] = ()

    def find_triggers(self, text: str) -> list[str]:
        """Return the unique trigger phrases that matched in `text`."""
        lowered = text.lower()
        hits: list[str] = []
        for pattern in self.trigger_patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                hits.append(pattern)
        return hits

    def has_mitigating_language(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            re.search(pattern, lowered, flags=re.IGNORECASE)
            for pattern in self.mitigating_patterns
        )


@dataclass
class RiskFinding:
    """A single concrete clause-level risk, ready for the report."""

    family: str
    severity: RiskSeverity
    description: str
    document_title: str
    chunk_id: str
    excerpt: str           # the cited text - the matched sentence, or the co-located sentence
    parent_excerpt: str    # the full paragraph for context
    score: float
    match_location: MatchLocation
    mitigations_detected: bool

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "severity": self.severity.value,
            "description": self.description,
            "document": self.document_title,
            "chunk_id": self.chunk_id,
            "excerpt": self.excerpt,
            "parent_excerpt": self.parent_excerpt,
            "retrieval_score": round(self.score, 4),
            "match_location": self.match_location.value,
            "mitigations_detected": self.mitigations_detected,
        }


@dataclass
class RiskAssessment:
    """The full audit result for one user query."""

    question: str
    findings: list[RiskFinding] = field(default_factory=list)
    retrieved_chunk_count: int = 0
    documents_scanned: list[str] = field(default_factory=list)

    @property
    def severity_summary(self) -> dict[str, int]:
        summary = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for finding in self.findings:
            summary[finding.severity.value] += 1
        return summary

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "documents_scanned": self.documents_scanned,
            "retrieved_chunk_count": self.retrieved_chunk_count,
            "severity_summary": self.severity_summary,
            "findings": [finding.to_dict() for finding in self.findings],
        }


# -- The risk library -----------------------------------------------------------
# Patterns are case-insensitive regex. I keep them in priority order: CRITICAL
# first, LOW last. The auditor does not depend on order, but it makes the
# source readable.

RISK_PATTERNS: tuple[RiskPattern, ...] = (
    RiskPattern(
        family="Binding arbitration & class-action waiver",
        severity=RiskSeverity.CRITICAL,
        description=(
            "Forces disputes into individual binding arbitration and blocks "
            "class actions. Often strips users of meaningful recourse."
        ),
        trigger_patterns=(
            r"binding.{0,20}arbitration",
            r"waive.{0,20}class.{0,5}action",
            r"class.{0,5}action.{0,20}waiver",
            r"individual.{0,5}arbitration",
            r"class\s+arbitration",
            r"no\s+class\s+proceedings",
        ),
        mitigating_patterns=(
            r"opt[- ]?out",
            r"right to (a )?jury",
        ),
    ),
    RiskPattern(
        family="Perpetual, irrevocable content license",
        severity=RiskSeverity.HIGH,
        description=(
            "User grants the service a permanent, royalty-free, irrevocable "
            "license to uploaded content - effectively transferring ownership."
        ),
        trigger_patterns=(
            r"perpetual.{0,15}irrevocable",
            r"irrevocable.{0,15}license",
            r"royalty[- ]?free.{0,30}perpetual",
            r"sublicense.{0,30}third part",
        ),
    ),
    RiskPattern(
        family="Unilateral right to change terms",
        severity=RiskSeverity.HIGH,
        description=(
            "Provider can change the agreement at any time, in their sole "
            "discretion, without notice - and continued use equals consent."
        ),
        trigger_patterns=(
            r"sole discretion",
            r"at any time.{0,30}without prior notice",
            r"right to (modify|amend|change).{0,40}at any time",
            r"continued use.{0,30}constitutes.{0,15}acceptance",
            r"effective immediately upon posting",
        ),
        mitigating_patterns=(
            r"30[- ]day notice",
            r"prior notice",
            r"material change",
        ),
    ),
    RiskPattern(
        family="Broad data selling & third-party sharing",
        severity=RiskSeverity.CRITICAL,
        description=(
            "Personal information is sold, rented, or shared with advertising "
            "partners, data brokers, or unaffiliated third parties."
        ),
        trigger_patterns=(
            r"sell.{0,30}personal (data|information)",
            r"share.{0,30}(advertising|marketing) (partners|networks)",
            r"data brokers",
            r"third[- ]party (advertis|partners).{0,30}targeted",
            r"sell,?\s*rent,?\s*lease",
        ),
    ),
    RiskPattern(
        family="Aggressive tracking & device fingerprinting",
        severity=RiskSeverity.HIGH,
        description=(
            "Cookies, tracking pixels, advertising IDs, and device "
            "fingerprinting are used to build detailed behavioural profiles."
        ),
        trigger_patterns=(
            r"tracking pixels?",
            r"device[- ]fingerprint",
            r"advertising ids?",
            r"web beacons?",
            r"session[- ]replay",
            r"click[- ]stream analytics",
        ),
    ),
    RiskPattern(
        family="Unilateral account termination & data deletion",
        severity=RiskSeverity.HIGH,
        description=(
            "Provider can suspend, terminate, or delete the account and all "
            "data at any time, with or without cause or notice."
        ),
        trigger_patterns=(
            r"(suspend|terminate).{0,30}at any time",
            r"with or without cause",
            r"permanently delete.{0,30}(account|content|data)",
            r"sole discretion.{0,40}(suspend|terminate|delete)",
        ),
    ),
    RiskPattern(
        family="Full liability disclaimer",
        severity=RiskSeverity.HIGH,
        description=(
            "Service is provided 'as-is' with no warranties, and the provider "
            "disclaims nearly all liability for damages including data loss."
        ),
        trigger_patterns=(
            r"as[- ]is.{0,15}as[- ]available",
            r"no warranties? of any kind",
            r"in no event shall.{0,80}be liable",
            r"disclaim.{0,20}all liability",
        ),
    ),
    RiskPattern(
        family="User-side indemnification",
        severity=RiskSeverity.MEDIUM,
        description=(
            "User is required to defend and indemnify the provider for any "
            "claim arising from the user's use of the service."
        ),
        trigger_patterns=(
            r"you (agree|shall) to indemnify",
            r"indemnify,? defend,? and hold harmless",
            r"defend,? indemnify,? and hold",
        ),
    ),
    RiskPattern(
        family="Cross-border data transfer with weak safeguards",
        severity=RiskSeverity.MEDIUM,
        description=(
            "User data is transferred to other countries, sometimes relying "
            "on standard contractual clauses with limited additional protection."
        ),
        trigger_patterns=(
            r"transfer.{0,30}out of your country",
            r"standard contractual clauses",
            r"transferred to.{0,20}(united states|any other country)",
        ),
    ),
    RiskPattern(
        family="Payment-card data exposure",
        severity=RiskSeverity.CRITICAL,
        description=(
            "Provider stores full payment card numbers, exposing users to "
            "breach risk. PCI-DSS compliant processors never do this."
        ),
        trigger_patterns=(
            r"store.{0,15}payment card",
            r"full card numbers?",
            r"card details are not (tokenized|encrypted)",
        ),
    ),
)


class RiskAuditor:
    """Applies the risk library to a list of retrieved chunks.

    Two-stage match: primary against the matched sentence, co-located
    against the rest of the parent paragraph. Co-located findings are
    demoted by one severity tier.
    """

    def __init__(self, patterns: Iterable[RiskPattern] = RISK_PATTERNS) -> None:
        self._patterns = tuple(patterns)

    def audit(self, question: str, hits: list[RetrievedChunk]) -> RiskAssessment:
        """Score every retrieved chunk against every risk pattern."""
        documents_scanned = sorted({hit.chunk.document_title for hit in hits})
        assessment = RiskAssessment(
            question=question,
            retrieved_chunk_count=len(hits),
            documents_scanned=documents_scanned,
        )

        for hit in hits:
            chunk = hit.chunk
            primary_findings = self._match_primary(chunk, hit.score)
            co_located_findings = self._match_co_located(chunk, hit.score)
            assessment.findings.extend(primary_findings)
            assessment.findings.extend(co_located_findings)

        # Highest-severity findings first; ties broken by retrieval score.
        assessment.findings.sort(
            key=lambda finding: (-finding.severity.rank, -finding.score)
        )
        _LOGGER.info(
            "Audit produced %d finding(s) for question: %s",
            len(assessment.findings),
            question,
        )
        return assessment

    # -- Helpers --------------------------------------------------------------

    def _match_primary(self, chunk, score: float) -> list[RiskFinding]:
        """Run patterns against the matched sentence. High-confidence path."""
        return self._build_findings(
            text=chunk.text,
            chunk=chunk,
            score=score,
            match_location=MatchLocation.PRIMARY,
            parent_excerpt=chunk.parent_text,
        )

    def _match_co_located(self, chunk, score: float) -> list[RiskFinding]:
        """Run patterns against the parent paragraph, excluding the matched sentence.

        This is how we surface "your matched sentence was about X, but the
        same paragraph also contains a Y risk you should know about".
        """
        # Strip the matched sentence out of the parent text so we don't
        # double-count the primary match. The parent is sentences joined
        # by a single space, so a substring check is good enough.
        parent_without_match = chunk.parent_text
        if chunk.text and chunk.text in parent_without_match:
            parent_without_match = parent_without_match.replace(chunk.text, " ")
        parent_without_match = " ".join(parent_without_match.split())
        if not parent_without_match:
            return []

        findings = self._build_findings(
            text=parent_without_match,
            chunk=chunk,
            score=score * 0.6,  # co-located matches are inherently weaker
            match_location=MatchLocation.CO_LOCATED,
            parent_excerpt=chunk.parent_text,
        )
        # Demote by one tier because the match is one sentence removed
        # from the user's actual question.
        demoted: list[RiskFinding] = []
        for finding in findings:
            demoted.append(
                RiskFinding(
                    family=finding.family,
                    severity=self._demote(finding.severity),
                    description=finding.description,
                    document_title=finding.document_title,
                    chunk_id=finding.chunk_id,
                    excerpt=finding.excerpt,
                    parent_excerpt=finding.parent_excerpt,
                    score=finding.score,
                    match_location=MatchLocation.CO_LOCATED,
                    mitigations_detected=finding.mitigations_detected,
                )
            )
        return demoted

    def _build_findings(
        self,
        text: str,
        chunk,
        score: float,
        match_location: MatchLocation,
        parent_excerpt: str,
    ) -> list[RiskFinding]:
        findings: list[RiskFinding] = []
        for pattern in self._patterns:
            triggers = pattern.find_triggers(text)
            if not triggers:
                continue
            mitigated = pattern.has_mitigating_language(text)
            effective_severity = self._effective_severity(pattern.severity, mitigated)
            findings.append(
                RiskFinding(
                    family=pattern.family,
                    severity=effective_severity,
                    description=pattern.description,
                    document_title=chunk.document_title,
                    chunk_id=chunk.chunk_id,
                    excerpt=self._build_excerpt(text),
                    parent_excerpt=parent_excerpt,
                    score=score,
                    match_location=match_location,
                    mitigations_detected=mitigated,
                )
            )
        return findings

    @staticmethod
    def _effective_severity(
        base: RiskSeverity, mitigated: bool
    ) -> RiskSeverity:
        """Demote one tier if mitigating language is detected in the same chunk."""
        if not mitigated:
            return base
        ladder = {
            RiskSeverity.CRITICAL: RiskSeverity.HIGH,
            RiskSeverity.HIGH: RiskSeverity.MEDIUM,
            RiskSeverity.MEDIUM: RiskSeverity.LOW,
            RiskSeverity.LOW: RiskSeverity.LOW,
        }
        return ladder[base]

    @staticmethod
    def _demote(severity: RiskSeverity) -> RiskSeverity:
        """Drop a severity by one tier unconditionally."""
        ladder = {
            RiskSeverity.CRITICAL: RiskSeverity.HIGH,
            RiskSeverity.HIGH: RiskSeverity.MEDIUM,
            RiskSeverity.MEDIUM: RiskSeverity.LOW,
            RiskSeverity.LOW: RiskSeverity.LOW,
        }
        return ladder[severity]

    @staticmethod
    def _build_excerpt(text: str, max_length: int = 320) -> str:
        """Truncate text to a human-readable excerpt, with ellipsis if needed."""
        collapsed = " ".join(text.split())
        if len(collapsed) <= max_length:
            return collapsed
        return collapsed[: max_length - 1].rstrip() + "…"
