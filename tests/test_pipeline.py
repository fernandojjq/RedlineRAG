"""End-to-end smoke test for the RedlineRAG pipeline.

This test runs the full pipeline against the auto-generated mock corpus
and asserts that:

  1. The pipeline builds a vector store without raising.
  2. A known "trap" query returns at least one CRITICAL finding.
  3. An unrelated query returns either no findings or only low-severity
     findings, proving the similarity floor works.
  4. The mock generator kicks in automatically when the raw dir is empty.
  5. (Regression tests for the QA pass) The "perpetual irrevocable
     content license" query is correctly classified as a content-licence
     risk, not a data-selling risk.
  6. (Regression tests for the QA pass) The "right to change terms"
     query is correctly classified as a unilateral-change risk, not an
     account-termination risk.
  7. Co-located findings are demoted by one tier.
  8. The reranker promotes chunks that share more exact terms with the
     query over chunks that only share statistical n-gram weight.

The test uses a temporary copy of the project tree so it never mutates
the user's real data directories.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from src.chunking.text_splitter import Chunk
from src.generation.auditor import MatchLocation, RiskSeverity
from src.pipeline.orchestrator import RagPipeline
from src.retrieval.reranker import TokenOverlapReranker
from src.retrieval.retriever import RetrievedChunk
from src.utils.config import PipelineConfig


@pytest.fixture()
def isolated_workspace() -> PipelineConfig:
    """Build a PipelineConfig whose directories all live in a temp dir."""
    tmp_root = Path(tempfile.mkdtemp(prefix="redline_rag_test_"))
    config = PipelineConfig(
        raw_documents_dir=tmp_root / "raw",
        mock_documents_dir=tmp_root / "mock",
        vector_store_dir=tmp_root / "vector_store",
        auto_generate_mocks=True,
    )
    config.ensure_directories()
    yield config
    shutil.rmtree(tmp_root, ignore_errors=True)


def test_pipeline_runs_end_to_end(isolated_workspace: PipelineConfig) -> None:
    """Full happy path: ingest -> index -> ask."""
    pipeline = RagPipeline(isolated_workspace)
    documents = pipeline.ingest()
    assert documents, "Mock generator should have produced at least one document."

    chunks_indexed = pipeline.index(force_rebuild=True)
    assert chunks_indexed > 0, "Indexer should have produced chunks."

    result = pipeline.ask("binding arbitration and class action waiver")
    assert result.assessment.findings, "Expected at least one finding for an obvious trap."
    severities = {finding.severity.value for finding in result.assessment.findings}
    assert "CRITICAL" in severities or "HIGH" in severities, (
        f"Expected CRITICAL or HIGH finding, got severities: {severities}"
    )


def test_pipeline_handles_repeat_runs(isolated_workspace: PipelineConfig) -> None:
    """The pipeline should be re-runnable without raising on a fresh state."""
    pipeline = RagPipeline(isolated_workspace)
    pipeline.ingest()
    pipeline.index(force_rebuild=True)
    pipeline.ask("data sharing with third parties")
    result = pipeline.ask("data sharing with third parties")
    assert result.chunks_indexed > 0


def test_similarity_floor_blocks_garbage_query(isolated_workspace: PipelineConfig) -> None:
    """A nonsense query should not produce spurious hits."""
    pipeline = RagPipeline(isolated_workspace)
    pipeline.ingest()
    pipeline.index(force_rebuild=True)

    result = pipeline.ask("the quick brown fox jumps over the lazy dog")
    assert result.assessment.retrieved_chunk_count == 0


def test_ingest_fails_cleanly_when_no_mocks(isolated_workspace: PipelineConfig) -> None:
    """Disabling auto-mock and providing no input should raise, not silently pass."""
    config = isolated_workspace.model_copy(update={"auto_generate_mocks": False})
    config.raw_documents_dir.mkdir(parents=True, exist_ok=True)
    for f in config.raw_documents_dir.iterdir():
        f.unlink()
    pipeline = RagPipeline(config)
    with pytest.raises(RuntimeError):
        pipeline.ingest()


# ----------------------------------------------------------------------------
# Regression tests for the QA-pass issues.
# Each test reproduces a real bug the user found and asserts the fix.
# ----------------------------------------------------------------------------


def test_content_license_query_is_not_data_selling(isolated_workspace: PipelineConfig) -> None:
    """Regression: 'perpetual irrevocable content license' must not be tagged
    as a 'Broad data selling' finding on the primary match.

    The chunk that scores highest for this query contains the words
    'perpetual', 'irrevocable', and 'license' (the content-licence trap).
    The 'data selling' pattern used to fire on the same chunk because the
    paragraph also mentions advertising partners. With sentence-level
    indexing, the matched sentence is about the licence, not the
    advertising, so the primary finding is the licence risk only.
    """
    pipeline = RagPipeline(isolated_workspace)
    pipeline.ingest()
    pipeline.index(force_rebuild=True)

    result = pipeline.ask("perpetual irrevocable content license")
    assert result.assessment.findings, "Expected at least one finding."

    primary_families = {
        finding.family
        for finding in result.assessment.findings
        if finding.match_location == MatchLocation.PRIMARY
    }
    assert "Perpetual, irrevocable content license" in primary_families, (
        f"Primary match must include the content-licence family. "
        f"Got: {primary_families}"
    )
    # The data-selling family is allowed to appear as a CO-LOCATED finding
    # (the same paragraph in mock_social_network talks about advertising
    # AND licensing), but it must NOT be a primary finding for this query.
    data_selling_primary = [
        finding
        for finding in result.assessment.findings
        if finding.match_location == MatchLocation.PRIMARY
        and finding.family == "Broad data selling & third-party sharing"
    ]
    assert not data_selling_primary, (
        "Data-selling family must not be a PRIMARY match for a query "
        "about content licensing."
    )


def test_unilateral_change_query_is_not_account_termination(
    isolated_workspace: PipelineConfig,
) -> None:
    """Regression: 'right to change terms without notice' must produce a
    primary 'Unilateral right to change terms' finding, not be mislabelled
    as 'Unilateral account termination & data deletion'.

    The previous version scanned the whole chunk and picked whichever
    pattern fired first; the matched sentence about amending terms would
    get tagged as 'account termination' because the paragraph also
    mentioned suspend/terminate. The fix: pattern-match against the
    matched sentence, not the whole chunk.

    Note: with sentence-level chunking, each clause is its own sentence.
    The query legitimately matches BOTH a "change terms" sentence and a
    "terminate access" sentence in the mock corpus, because they share
    "at any time" and "without notice". The key invariant we test is
    that the sentence that is *primarily about* changing terms gets the
    'Unilateral right to change terms' family label - not 'Account
    termination'.
    """
    pipeline = RagPipeline(isolated_workspace)
    pipeline.ingest()
    pipeline.index(force_rebuild=True)

    result = pipeline.ask("right to change terms without notice")
    assert result.assessment.findings, "Expected at least one finding."

    primary_findings = [
        finding
        for finding in result.assessment.findings
        if finding.match_location == MatchLocation.PRIMARY
    ]
    assert any(
        finding.family == "Unilateral right to change terms"
        for finding in primary_findings
    ), (
        f"Primary matches must include 'Unilateral right to change terms'. "
        f"Got: {[f.family for f in primary_findings]}"
    )

    # For every primary finding, the cited text must contain language
    # consistent with the labeled family. This is the real bug we are
    # guarding against: a sentence about "amend these Terms" being
    # labeled as "account termination".
    for finding in primary_findings:
        excerpt_lower = finding.excerpt.lower()
        if finding.family == "Unilateral right to change terms":
            # The cited text should actually talk about modifying/amending terms.
            assert any(
                keyword in excerpt_lower
                for keyword in (
                    "modify", "amend", "change", "update", "revised",
                    "terms", "sole discretion", "prior notice",
                )
            ), (
                f"Finding labeled 'Unilateral right to change terms' has "
                f"unrelated cited text: {finding.excerpt!r}"
            )
        elif finding.family == "Unilateral account termination & data deletion":
            # The cited text should actually talk about suspending/terminating
            # access, not about changing terms.
            assert any(
                keyword in excerpt_lower
                for keyword in (
                    "suspend", "terminate", "delete", "account", "access",
                )
            ), (
                f"Finding labeled 'Unilateral account termination' has "
                f"unrelated cited text: {finding.excerpt!r}"
            )


def test_co_located_findings_are_demoted(isolated_workspace: PipelineConfig) -> None:
    """A co-located finding must be at least one severity tier below the
    equivalent primary finding."""
    pipeline = RagPipeline(isolated_workspace)
    pipeline.ingest()
    pipeline.index(force_rebuild=True)

    # A query that pulls a paragraph with multiple distinct risk families.
    result = pipeline.ask("data sharing with third parties")
    findings = result.assessment.findings

    primary_families = {
        finding.family: finding.severity
        for finding in findings
        if finding.match_location == MatchLocation.PRIMARY
    }
    co_located_families = {
        finding.family: finding.severity
        for finding in findings
        if finding.match_location == MatchLocation.CO_LOCATED
    }

    # For every co-located family, the primary version of the same family
    # must exist and must be at least one tier more severe.
    _SEVERITY_ORDER = {
        RiskSeverity.LOW: 1,
        RiskSeverity.MEDIUM: 2,
        RiskSeverity.HIGH: 3,
        RiskSeverity.CRITICAL: 4,
    }
    for family, co_located_severity in co_located_families.items():
        if family in primary_families:
            assert (
                _SEVERITY_ORDER[primary_families[family]]
                > _SEVERITY_ORDER[co_located_severity]
            ), (
                f"Co-located severity for '{family}' ({co_located_severity}) "
                f"must be below primary severity ({primary_families[family]})."
            )


def test_reranker_promotes_exact_term_overlap() -> None:
    """The reranker should put a chunk that shares more exact terms with
    the query above a chunk that only shares statistical n-gram weight.
    """
    reranker = TokenOverlapReranker(alpha=0.5)

    # Two candidates. Both have moderate vector similarity. The first
    # shares 3 of 3 query tokens exactly; the second shares 0 of 3.
    hits = [
        _make_hit("binding arbitration class action waiver", vector_score=0.30),
        _make_hit("perpetual irrevocable license royalty free", vector_score=0.28),
    ]
    reranked = reranker.rerank("binding arbitration class action", hits)
    assert reranked, "Reranker should return at least one hit"
    # The arbitration chunk must come first because it shares exact
    # tokens with the query; the licence chunk has zero overlap.
    assert "arbitration" in reranked[0].chunk.text.lower()


def test_reranker_drops_low_overlap_candidates() -> None:
    """A candidate whose token overlap with the query is below the
    `min_overlap_ratio` must be dropped before scoring. This is the
    fix for the false-positive where a sentence shares one statistical
    n-gram with the query but is about a different topic.
    """
    reranker = TokenOverlapReranker(alpha=0.5, min_overlap_ratio=0.4)

    # Query has 4 meaningful tokens: "tracking", "pixels", "device",
    # "fingerprinting". The first candidate shares 4/4 = 1.0; the second
    # shares 1/4 = 0.25, below the 0.4 floor, so it must be dropped.
    hits = [
        _make_hit(
            "tracking pixels device fingerprinting technologies",
            vector_score=0.30,
        ),
        _make_hit(
            "share personal information advertising partners data brokers",
            vector_score=0.32,
        ),
    ]
    reranked = reranker.rerank("tracking pixels device fingerprinting", hits)
    assert len(reranked) == 1, (
        f"Expected only the on-topic candidate to survive; got {len(reranked)}"
    )
    assert "tracking" in reranked[0].chunk.text.lower()


def test_tracking_query_does_not_return_data_selling(
    isolated_workspace: PipelineConfig,
) -> None:
    """Regression for the QA-pass issue: 'tracking pixels and device
    fingerprinting' should NOT produce a 'Broad data selling' finding,
    even though the data-selling sentence shares the word 'device' with
    the query. The new min_overlap_ratio + key_signals combination
    filters the off-topic candidate.
    """
    pipeline = RagPipeline(isolated_workspace)
    pipeline.ingest()
    pipeline.index(force_rebuild=True)

    result = pipeline.ask("tracking pixels and device fingerprinting")

    families = {finding.family for finding in result.assessment.findings}
    assert "Aggressive tracking & device fingerprinting" in families, (
        f"Expected 'Aggressive tracking' finding. Got: {families}"
    )
    # The data-selling family must not appear as on-topic for this query.
    # It may still appear as off-topic if it slipped through, but the
    # audit system should not surface it as a confident result.
    on_topic_data_selling = [
        finding
        for finding in result.assessment.findings
        if finding.family == "Broad data selling & third-party sharing"
        and not finding.off_topic
    ]
    assert not on_topic_data_selling, (
        "Data-selling family must not be on-topic for a tracking query."
    )


def test_unilateral_change_pattern_covers_real_world_phrasings(
    isolated_workspace: PipelineConfig,
) -> None:
    """Regression for the QA-pass issue: queries phrased like real
    ToS language ('modify this Agreement', 'update this Agreement')
    must trigger the unilateral-change family.

    The mock data uses 'modify... these Terms' phrasing which already
    matched, but the new pattern additions ('modify this Agreement',
    'we may update this Agreement') should let the family also match
    real-world documents that use slightly different wording.
    """
    from src.generation.auditor import RiskAuditor
    from src.generation.auditor import RISK_PATTERNS

    # Find the unilateral-change pattern in the library.
    pattern = next(
        p for p in RISK_PATTERNS if p.family == "Unilateral right to change terms"
    )
    # These real-world phrasings should each trigger at least one
    # pattern AND have at least one matching key signal.
    real_world_clauses = [
        "We reserve the right to modify this Agreement at any time.",
        "We may update these Terms from time to time in our sole discretion.",
        "We may amend this Agreement at any time, with or without notice.",
        "We may change this Agreement at any time by posting the revised version.",
        "The updated Terms will be effective immediately upon posting.",
    ]
    for clause in real_world_clauses:
        assert pattern.find_triggers(clause), (
            f"Pattern must trigger on real-world phrasing: {clause!r}"
        )
        assert pattern.has_key_signal(clause), (
            f"Pattern must recognise the key signal in: {clause!r}"
        )


def test_off_topic_pattern_is_demoted() -> None:
    """If a pattern's triggers fire but no key signal is present in the
    text, the finding is marked off_topic and demoted one tier."""
    from src.generation.auditor import (
        RiskAuditor,
        RiskPattern,
        RiskSeverity,
    )

    # Build a pattern with key_signals that the test text does NOT contain.
    pattern = RiskPattern(
        family="Test family",
        severity=RiskSeverity.CRITICAL,
        description="Test",
        trigger_patterns=(r"share.{0,30}partners",),
        key_signals=("data brokers", "sell"),
    )
    auditor = RiskAuditor(patterns=[pattern])

    # Text triggers the regex ("share... partners") but has none of the
    # key signals ("data brokers", "sell").
    hit = _make_hit("We may share technical partners with our service.", 0.5)
    assessment = auditor.audit("test", [hit])
    assert len(assessment.findings) == 1
    finding = assessment.findings[0]
    assert finding.off_topic is True
    # CRITICAL -> demoted once to HIGH.
    assert finding.severity == RiskSeverity.HIGH


def test_absence_of_notice_is_not_mitigating() -> None:
    """Regression: the 'without prior notice' phrasing must NOT be
    treated as a mitigating promise. A clause that says 'we may change
    terms without prior notice' is the OPPOSITE of mitigation - it is
    the worst case. Earlier the mitigating pattern `r'prior notice'`
    matched the substring 'prior notice' in 'without prior notice',
    demoting severity and hiding the risk. The fix uses a negative
    lookbehind: `r'(?<!without )prior notice'`.
    """
    from src.generation.auditor import (
        RiskAuditor,
        RiskSeverity,
    )

    pattern = next(
        p for p in RiskAuditor()._patterns
        if p.family == "Unilateral right to change terms"
    )
    # "without prior notice" must NOT register as mitigation.
    assert not pattern.has_mitigating_language(
        "We may modify these terms at any time, without prior notice."
    ), "'without prior notice' should not be treated as a mitigating promise"
    # "we'll give 30 days notice" MUST register as mitigation.
    assert pattern.has_mitigating_language(
        "We will give you 30 days notice for any material change."
    ), "an actual notice promise should still be treated as mitigation"


def _make_hit(text: str, vector_score: float) -> RetrievedChunk:
    """Test helper: build a RetrievedChunk with a fake Chunk."""
    chunk = Chunk(
        chunk_id=f"test::{text[:8]}",
        document_id="test_doc",
        document_title="Test Doc",
        text=text,
        parent_text=text,
        position=0,
        metadata={},
    )
    return RetrievedChunk(chunk=chunk, score=vector_score)
