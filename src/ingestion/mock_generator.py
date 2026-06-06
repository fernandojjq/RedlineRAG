"""Synthesizes realistic Terms of Service and privacy policy samples.

When the user has no real documents to analyze, this module creates a small
corpus of mock agreements that look like the real thing and contain the
exact kinds of legal traps a real auditor would flag:

  * Forced binding arbitration with class-action waiver
  * Perpetual, irrevocable, royalty-free content licensing
  * Unilateral right to change terms without notice
  * Broad third-party data sharing / sale
  * Tracking pixels, advertising IDs, device fingerprinting
  * Cross-border data transfer with weak safeguards
  * Account suspension / data deletion at the provider's sole discretion
  * "As-is" / full liability disclaimer
  * Indemnification of the provider by the user

The point is not to ship realistic legal text - it is to make sure the
risk detection rules in the auditor are wired to *something* on a fresh
checkout, so the pipeline is observably working from the first run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


# -- Clause library -------------------------------------------------------------
# Each clause is intentionally written in the dense, legalistic register real
# ToS use. We use a few paraphrases per risk category so the index isn't
# trivially hashable by a single substring.

_DATA_SELLING_CLAUSES: tuple[str, ...] = (
    "We may share your personal information, including your contact details, "
    "browsing history, and device identifiers, with our advertising partners, "
    "data brokers, and affiliated marketing networks for targeted advertising "
    "and audience profiling purposes.",

    "By using the Service you grant us a fully paid-up, royalty-free, "
    "perpetual, and irrevocable license to use, reproduce, modify, and "
    "distribute any content you upload for any purpose, including commercial "
    "exploitation and sublicensing to third parties.",

    "We may sell, rent, lease, or otherwise transfer aggregated and de-identified "
    "user data, as well as granular behavioural and location data, to unaffiliated "
    "third parties for analytics, advertising, and product-improvement purposes.",
)

_BINDING_ARBITRATION_CLAUSES: tuple[str, ...] = (
    "Any dispute arising out of or relating to these Terms shall be resolved "
    "exclusively by binding individual arbitration administered by the American "
    "Arbitration Association. You hereby waive your right to participate in any "
    "class action, class arbitration, or other representative action.",

    "You and the Company agree that any controversy or claim shall be settled "
    "by confidential, final, and binding arbitration on an individual basis "
    "only. Class-wide proceedings, joinder, and consolidation are expressly "
    "disallowed.",
)

_UNILATERAL_CHANGE_CLAUSES: tuple[str, ...] = (
    "We reserve the right to modify, suspend, or discontinue the Service, or "
    "to amend these Terms, at any time, in our sole discretion, without prior "
    "notice to you. Your continued use of the Service constitutes acceptance "
    "of the modified Terms.",

    "We may update these Terms at any time by posting the revised version on "
    "our website. It is your responsibility to review the Terms periodically. "
    "The updated Terms will be effective immediately upon posting.",
)

_ACCOUNT_TERMINATION_CLAUSES: tuple[str, ...] = (
    "We may suspend, limit, or terminate your access to the Service at any "
    "time, with or without cause, and with or without notice. Upon termination "
    "we may permanently delete your account data, content, and any associated "
    "records without obligation to provide a backup.",

    "We reserve the right to delete or disable any user account, content, or "
    "data at our sole discretion, including but not limited to accounts we "
    "deem to be inactive, suspicious, or in violation of these Terms.",
)

_LIABILITY_DISCLAIMER_CLAUSES: tuple[str, ...] = (
    "THE SERVICE IS PROVIDED ON AN 'AS-IS' AND 'AS-AVAILABLE' BASIS WITHOUT "
    "WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT "
    "LIMITATION IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR "
    "PURPOSE, AND NON-INFRINGEMENT.",

    "TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL THE "
    "COMPANY, ITS AFFILIATES, OFFICERS, DIRECTORS, EMPLOYEES, AGENTS, OR "
    "LICENSORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, "
    "OR PUNITIVE DAMAGES, INCLUDING LOST PROFITS OR LOST DATA, ARISING OUT OF "
    "OR IN CONNECTION WITH YOUR USE OF THE SERVICE.",
)

_INDEMNIFICATION_CLAUSES: tuple[str, ...] = (
    "You agree to indemnify, defend, and hold harmless the Company and its "
    "affiliates from and against any and all claims, damages, obligations, "
    "losses, liabilities, costs, or expenses (including reasonable attorneys' "
    "fees) arising from your access to or use of the Service.",

    "You shall defend, indemnify, and hold the Company harmless from any "
    "third-party claim, demand, action, or proceeding arising out of or "
    "related to your breach of these Terms or your misuse of the Service.",
)

_TRACKING_CLAUSES: tuple[str, ...] = (
    "We, along with our third-party partners, automatically collect information "
    "about your device, browser type, IP address, advertising IDs, and "
    "interaction patterns through cookies, web beacons, tracking pixels, and "
    "device-fingerprinting technologies.",

    "We use session-replay, heat-mapping, and click-stream analytics tools to "
    "record your interactions with the Service, including mouse movements, "
    "keystroke patterns, and scroll behaviour, for product improvement.",
)

_DATA_TRANSFER_CLAUSES: tuple[str, ...] = (
    "Your information may be transferred to, stored, and processed in the "
    "United States or any other country in which we or our service providers "
    "operate. By using the Service you consent to the transfer of your "
    "information out of your country of residence.",

    "We rely on standard contractual clauses and other mechanisms approved by "
    "the European Commission for the transfer of personal data outside the "
    "European Economic Area, the United Kingdom, and Switzerland.",
)


@dataclass(frozen=True)
class _MockAgreement:
    """One self-contained mock agreement, with planted traps."""

    slug: str
    title: str
    company: str
    intro: str
    trap_clauses: tuple[str, ...]
    safe_clauses: tuple[str, ...]

    def render(self) -> str:
        """Render the agreement to a single plain-text document."""
        today = datetime.now(tz=timezone.utc).strftime("%B %d, %Y")
        header = (
            f"{self.title}\n"
            f"{self.company}\n"
            f"Last updated: {today}\n"
            f"{'=' * 72}\n\n"
        )
        body_paragraphs = [self.intro, *self.trap_clauses, *self.safe_clauses]
        body = "\n\n".join(paragraph_paragraph for paragraph_paragraph in body_paragraphs)
        return header + body + "\n"


# -- Three sample agreements, each with a different risk profile ---------------


def _social_network_agreement() -> _MockAgreement:
    return _MockAgreement(
        slug="mock_social_network",
        title="Terms of Service - Brightwave Social",
        company="Brightwave Social, Inc.",
        intro=(
            "Welcome to Brightwave Social. These Terms of Service govern your "
            "access to and use of our social-networking platform, including "
            "our website, mobile applications, and related services."
        ),
        trap_clauses=(
            *_DATA_SELLING_CLAUSES,
            *_TRACKING_CLAUSES,
            *_UNILATERAL_CHANGE_CLAUSES,
            *_ACCOUNT_TERMINATION_CLAUSES,
            *_LIABILITY_DISCLAIMER_CLAUSES,
        ),
        safe_clauses=(
            "We implement reasonable administrative, technical, and physical "
            "safeguards designed to protect the security of your personal "
            "information against unauthorized access, disclosure, or misuse.",
            "If you wish to delete your account, you may do so at any time "
            "from the settings page. We will process your deletion request "
            "within thirty (30) days of receipt.",
        ),
    )


def _marketplace_agreement() -> _MockAgreement:
    return _MockAgreement(
        slug="mock_marketplace",
        title="User Agreement - Quickbid Marketplace",
        company="Quickbid Holdings Ltd.",
        intro=(
            "These User Agreement terms apply to buyers, sellers, and "
            "browsers using Quickbid Marketplace. By listing, bidding on, or "
            "purchasing items you agree to be bound by these terms in full."
        ),
        trap_clauses=(
            *_BINDING_ARBITRATION_CLAUSES,
            *_INDEMNIFICATION_CLAUSES,
            *_DATA_TRANSFER_CLAUSES,
            *_LIABILITY_DISCLAIMER_CLAUSES,
        ),
        safe_clauses=(
            "We offer a fourteen (14) day money-back guarantee on eligible "
            "purchases, subject to the seller's individual return policy.",
            "We do not store full payment card numbers on our servers. Card "
            "details are tokenized and processed by our PCI-DSS Level 1 "
            "payment processor.",
        ),
    )


def _cloud_saas_agreement() -> _MockAgreement:
    return _MockAgreement(
        slug="mock_cloud_saas",
        title="Cloud Service Agreement - Nimblecloud",
        company="Nimblecloud Technologies, Inc.",
        intro=(
            "This Cloud Service Agreement governs your subscription-based use "
            "of Nimblecloud's hosted infrastructure, data-processing, and "
            "storage services."
        ),
        trap_clauses=(
            *_BINDING_ARBITRATION_CLAUSES,
            *_DATA_TRANSFER_CLAUSES,
            *_INDEMNIFICATION_CLAUSES,
            *_UNILATERAL_CHANGE_CLAUSES,
        ),
        safe_clauses=(
            "We publish a public status page with real-time availability and "
            "historical uptime metrics for all production services.",
            "Customer data is encrypted at rest using AES-256 and in transit "
            "using TLS 1.2 or higher, with optional customer-managed "
            "encryption keys (CMEK) available on Enterprise plans.",
        ),
    )


_MOCK_LIBRARY: tuple[_MockAgreement, ...] = (
    _social_network_agreement(),
    _marketplace_agreement(),
    _cloud_saas_agreement(),
)


class MockTosGenerator:
    """Writes the canned mock library to disk on demand."""

    def __init__(self, output_directory: Path) -> None:
        self._output_directory = Path(output_directory)

    def generate(self, force: bool = False) -> list[Path]:
        """Materialize the mock library on disk.

        Returns the list of file paths actually written. If files already
        exist and `force` is False, the existing files are kept and the
        generator is a no-op.
        """
        self._output_directory.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for agreement in _MOCK_LIBRARY:
            target = self._output_directory / f"{agreement.slug}.txt"
            if target.exists() and not force:
                _LOGGER.debug("Mock already present, skipping: %s", target.name)
                continue
            target.write_text(agreement.render(), encoding="utf-8")
            written.append(target)
            _LOGGER.info("Wrote mock ToS: %s", target.name)
        return written
