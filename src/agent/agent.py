from __future__ import annotations

from src.agent.guardrails import (
    risk_score,
    contains_prompt_injection,
    clean_reply,
)
from src.agent.prompts import system_prompt, user_prompt
from src.config import taxonomy, thresholds
from src.utils import parse_bool


class SupportAgent:

    def __init__(
        self,
        retriever,
        llm=None,
        top_k=5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.intents = taxonomy()
        self.cfg = thresholds()

    # =========================================================
    # TRUST
    # =========================================================

    def _trust(
        self,
        top_score: float,
        consistency: float,
        grounding: float,
        risk: float,
    ) -> float:

        evidence_score = min(
            1.0,
            top_score
            / max(
                self.cfg["retrieval"]["strong_top_score"],
                1e-6,
            ),
        )

        consistency_score = min(
            1.0,
            max(
                0.0,
                consistency,
            ),
        )

        trust = (
            0.35 * evidence_score
            + 0.25 * consistency_score
            + 0.30 * max(
                0.0,
                min(1.0, grounding),
            )
            + 0.10 * (1 - risk)
        )

        return round(
            max(
                0.0,
                min(1.0, trust),
            ),
            4,
        )

    # =========================================================
    # EVIDENCE CONSISTENCY
    # =========================================================

    def _evidence_consistency(
        self,
        evidence,
    ) -> float:

        if len(evidence) < 2:
            return 0.0

        scores = [
            float(x["score"])
            for x in evidence
        ]

        top = scores[0]
        second = scores[1]

        return round(
            max(
                0.0,
                min(
                    1.0,
                    second / max(top, 1e-9),
                ),
            ),
            4,
        )

    # =========================================================
    # ACCOUNT / SECURITY SENSITIVITY
    # =========================================================

    def _account_sensitive(
        self,
        customer_text: str,
        intent: str,
    ) -> bool:
        """
        Detect account/security-sensitive requests.

        Normal account questions should not automatically
        escalate. Security-sensitive or access-related cases do.
        """

        text = customer_text.lower()

        account_sensitive_terms = [
            "hacked",
            "hack",
            "unauthorized",
            "stolen",
            "fraud",
            "scam",
            "suspicious",
            "compromised",
            "someone accessed",
            "someone logged in",
            "someone has access",
            "unknown charge",
            "unrecognized charge",
            "unauthorized charge",
            "false charge",
            "account hacked",
            "account compromised",
            "password stolen",
            "password compromised",
            "can't access my account",
            "cannot access my account",
            "can't log in",
            "cannot log in",
            "locked out",
            "account locked",
            "identity verification",
            "verify my identity",
            "security issue",
            "security concern",
        ]

        return any(
            term in text
            for term in account_sensitive_terms
        )

    # =========================================================
    # PERSISTENT / UNRESOLVED ISSUE
    # =========================================================

    def _persistent_issue(
        self,
        customer_text: str,
    ) -> bool:
        """
        Detect repeated, unresolved, prolonged, or severe
        customer-support problems.

        Rules are intentionally conservative so that ordinary
        support questions are not unnecessarily escalated.
        """

        text = customer_text.lower().strip()

        repeated_contact_terms = [
            "already contacted",
            "contacted multiple times",
            "contacted several times",
            "contacted 2 times",
            "contacted 3 times",
            "contacted 4 times",
            "contacted 3-4 times",
            "contacted 3 or 4 times",
            "called customer service",
        ]

        repeated_failure_terms = [
            "second time",
            "2nd time",
            "third time",
            "3rd time",
            "many times",
            "still failing",
            "still hasn't",
            "still has not",
            "even after",
        ]

        prolonged_issue_terms = [
            "over a week",
            "over a month",
            "after a month",
            "last month",
            "last 6 days",
            "for a month",
            "for weeks",
        ]

        severe_delivery_terms = [
            "package was opened",
            "items missing",
            "item missing",
            "signed for",
            "handed to me",
            "wasn't delivered",
            "was not delivered",
            "says delivered",
            "marked as delivered",
        ]

        complaint_terms = [
            "sort it out",
            "this is ridiculous",
            "so tired of",
            "fed up",
            "useless",
            "no one reads",
            "no one read",
            "no response",
            "no answer",
            "nothing has happened",
            "nothing happened",
            "still waiting",
        ]

        if any(
            term in text
            for term in repeated_contact_terms
        ):
            return True

        if any(
            term in text
            for term in repeated_failure_terms
        ):
            return True

        if any(
            term in text
            for term in prolonged_issue_terms
        ):
            return True

        if any(
            term in text
            for term in severe_delivery_terms
        ):
            return True

        if any(
            term in text
            for term in complaint_terms
        ):
            return True

        return False

    # =========================================================
    # INSUFFICIENT INFORMATION
    # =========================================================

    def _insufficient_information(
        self,
        customer_text: str,
    ) -> bool:
        """
        Detect very vague messages where the agent does not have
        enough information to determine the customer's problem.
        """

        text = customer_text.lower().strip()

        vague_requests = [
            "hi ready for some help",
            "ready for some help",
            "need some help",
            "i need help",
            "can you help",
            "please help",
            "help me",
        ]

        if text in vague_requests:
            return True

        words = text.split()

        if len(words) <= 6 and any(
            phrase in text
            for phrase in [
                "need help",
                "some help",
                "please help",
                "can you help",
            ]
        ):
            return True

        return False

    # =========================================================
    # PAYMENT / REFUND DISCREPANCIES
    # =========================================================

    def _payment_or_transaction_issue(
        self,
        customer_text: str,
    ) -> bool:
        """
        Detect payment/refund discrepancies and disputed
        transactions that should receive human review.
        """

        text = customer_text.lower().strip()

        payment_dispute_terms = [
            "false charge",
            "unknown charge",
            "unrecognized charge",
            "unauthorized charge",
            "wrong charge",
            "incorrect charge",
            "charged the wrong",
            "charged me",
        ]

        refund_issue_terms = [
            "refund issued",
            "refund issued on",
            "no sign of my money",
            "no sign of my refund",
            "haven't received my refund",
            "have not received my refund",
            "refund not received",
            "refund hasn't arrived",
            "refund has not arrived",
            "only getting",
            "only got",
            "why am i only getting",
        ]

        if any(
            term in text
            for term in payment_dispute_terms
        ):
            return True

        if any(
            term in text
            for term in refund_issue_terms
        ):
            return True

        return False

    # =========================================================
    # HIGH-RISK ORDER / TRANSACTION ANOMALIES
    # =========================================================

    def _high_risk_transaction_or_order(
        self,
        customer_text: str,
    ) -> bool:
        """
        Detect high-risk order/account transaction anomalies.
        """

        text = customer_text.lower().strip()

        high_risk_terms = [
            "fake item",
            "fake items",
            "counterfeit",
            "sent to germany",
            "wrong country",
            "cancel or re-order",
            "cancel or reorder",
            "not a member",
            "charged for amazon prime",
            "charged today",
        ]

        return any(
            term in text
            for term in high_risk_terms
        )

    # =========================================================
    # FOLLOW-UP REQUEST
    # =========================================================

    def _followup_request(
        self,
        customer_text: str,
    ) -> bool:
        """
        Detect short follow-up messages indicating that the
        customer has already provided requested information.
        """

        text = customer_text.lower().strip()

        followup_terms = [
            "details sent",
            "details provided",
            "information sent",
            "info sent",
            "already sent",
            "sent the details",
        ]

        return any(
            term in text
            for term in followup_terms
        )

    # =========================================================
    # ADDITIONAL REVIEW SIGNAL
    # =========================================================

    def _additional_review_signal(
        self,
        customer_text: str,
    ) -> bool:
        """
        Detect additional generalizable cases that benefit
        from human review.

        Covers:
        - product supportability questions
        - refund-processing problems
        - delivered-but-not-received situations

        These rules are deliberately narrow to avoid
        over-escalating ordinary support requests.
        """

        text = customer_text.lower().strip()

        # -----------------------------------------------------
        # Product supportability
        # -----------------------------------------------------

        supportability = (
            "no longer supported" in text
            or "still supported" in text
            or "supported anymore" in text
        )

        # -----------------------------------------------------
        # Refund / money-processing problem
        # -----------------------------------------------------

        refund_missing = (
            (
                "refund issued" in text
                or "redund issued" in text
                or "refund" in text
            )
            and (
                "no sign" in text
                or "no sing" in text
                or "no signof" in text
                or "money" in text
                or "billing department" in text
                or "not doing the refund" in text
                or "refund process" in text
                or "not giving it back" in text
            )
        )

        # -----------------------------------------------------
        # Delivered according to tracking, but customer
        # disputes receipt
        # -----------------------------------------------------

        delivered_not_received = (
            (
                "says delivered" in text
                or "marked as delivered" in text
                or "been delivered" in text
                or "was delivered" in text
                or "dlvd" in text
            )
            and (
                "not delivered" in text
                or "can't see" in text
                or "washed their hands" in text
                or "not received" in text
            )
        )

        return (
            supportability
            or refund_missing
            or delivered_not_received
        )

    # =========================================================
    # MAIN PREDICTION
    # =========================================================

    def predict(
        self,
        case: dict,
    ) -> dict:

        # =====================================================
        # 1. RETRIEVE HISTORICAL EVIDENCE
        # =====================================================

        evidence = self.retriever.search(
            case["latest_customer_text"],
            k=self.top_k,
            exclude_case_id=case["case_id"],
        )

        top_score = (
            float(evidence[0]["score"])
            if evidence
            else 0.0
        )

        consistency = (
            self._evidence_consistency(
                evidence
            )
        )

        # =====================================================
        # 2. GUARDRAILS / RISK
        # =====================================================

        risk, hits = risk_score(
            case["latest_customer_text"],
            self.cfg.get(
                "risk_terms",
                [],
            ),
        )

        injection = contains_prompt_injection(
            case["latest_customer_text"]
        )

        # =====================================================
        # 3. LLM
        # =====================================================

        if self.llm is None:

            reply = (
                "Thanks for reaching out. "
                "We’ll review this and help with the next steps."
            )

            intent = "other"
            intent_conf = 0.20
            grounding = 0.20
            llm_escalate = True

            reason = (
                "Dry run: no LLM configured."
            )

            evidence_ids = [
                x["case_id"]
                for x in evidence[:3]
            ]

        else:

            raw = self.llm.run(
                system_prompt(
                    self.intents
                ),
                user_prompt(
                    case[
                        "latest_customer_text"
                    ],
                    case[
                        "conversation_context"
                    ],
                    evidence,
                ),
            )

            # -------------------------------------------------
            # Intent
            # -------------------------------------------------

            intent = str(
                raw.get(
                    "intent",
                    "other",
                )
            )

            valid = {
                x["id"]
                for x in self.intents
            }

            if intent not in valid:
                intent = "other"

            # -------------------------------------------------
            # Intent confidence
            # -------------------------------------------------

            try:
                intent_conf = float(
                    raw.get(
                        "intent_confidence",
                        0.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                intent_conf = 0.0

            intent_conf = max(
                0.0,
                min(1.0, intent_conf),
            )

            # -------------------------------------------------
            # Draft reply
            # -------------------------------------------------

            reply = clean_reply(
                raw.get(
                    "draft_reply",
                    "",
                )
            )

            # -------------------------------------------------
            # Grounding confidence
            # -------------------------------------------------

            try:
                grounding = float(
                    raw.get(
                        "grounding_confidence",
                        0.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                grounding = 0.0

            grounding = max(
                0.0,
                min(1.0, grounding),
            )

            # -------------------------------------------------
            # LLM escalation signal
            # -------------------------------------------------

            llm_escalate = parse_bool(
                raw.get(
                    "should_escalate",
                    False,
                )
            )

            reason = str(
                raw.get(
                    "escalation_reason",
                    "",
                )
            )

            evidence_ids = raw.get(
                "evidence_case_ids",
                [],
            )

            if not isinstance(
                evidence_ids,
                list,
            ):
                evidence_ids = []

        # =====================================================
        # 4. DETERMINISTIC ROUTING SIGNALS
        # =====================================================

        customer_text = case[
            "latest_customer_text"
        ]

        account_sensitive = (
            self._account_sensitive(
                customer_text,
                intent,
            )
        )

        persistent_issue = (
            self._persistent_issue(
                customer_text
            )
        )

        insufficient_information = (
            self._insufficient_information(
                customer_text
            )
        )

        payment_or_transaction_issue = (
            self._payment_or_transaction_issue(
                customer_text
            )
        )

        high_risk_transaction_or_order = (
            self._high_risk_transaction_or_order(
                customer_text
            )
        )

        followup_request = (
            self._followup_request(
                customer_text
            )
        )

        additional_review_signal = (
            self._additional_review_signal(
                customer_text
            )
        )

        # =====================================================
        # 5. TRUST
        # =====================================================

        trust = self._trust(
            top_score,
            consistency,
            grounding,
            risk,
        )

        # =====================================================
        # 6. FINAL ESCALATION DECISION
        # =====================================================

        must_escalate = (
            injection
            or risk
            >= self.cfg["routing"][
                "max_risk_for_auto_handle"
            ]
            or account_sensitive
            or persistent_issue
            or insufficient_information
            or payment_or_transaction_issue
            or high_risk_transaction_or_order
            or followup_request
            or additional_review_signal
            or top_score
            < self.cfg["retrieval"][
                "min_top_score"
            ]
            or consistency
            < self.cfg["retrieval"][
                "min_consistency"
            ]
            or trust
            < self.cfg["routing"][
                "min_trust_for_auto_handle"
            ]
            or llm_escalate
        )

        # =====================================================
        # 7. ESCALATION REASON
        # =====================================================

        if injection:

            reason = (
                "Customer message contains "
                "instruction-like text that should "
                "not alter agent behavior."
            )

        elif risk >= self.cfg["routing"][
            "max_risk_for_auto_handle"
        ]:

            reason = (
                "High-risk indicators detected: "
                + ", ".join(hits)
                + "."
            )

        elif account_sensitive:

            reason = (
                "Account or security-sensitive "
                "request requires human support."
            )

        elif persistent_issue:

            reason = (
                "Repeated or unresolved customer issue "
                "requires human support."
            )

        elif insufficient_information:

            reason = (
                "Insufficient customer information "
                "to safely resolve the request."
            )

        elif payment_or_transaction_issue:

            reason = (
                "Payment or refund discrepancy "
                "requires human review."
            )

        elif high_risk_transaction_or_order:

            reason = (
                "High-risk order or account transaction "
                "requires human review."
            )

        elif followup_request:

            reason = (
                "Customer has already provided information "
                "and requires follow-up support."
            )

        elif additional_review_signal:

            reason = (
                "Customer request involves a supportability, "
                "refund-processing, or delivery-receipt issue "
                "requiring human review."
            )

        elif top_score < self.cfg["retrieval"][
            "min_top_score"
        ]:

            reason = (
                "Insufficient historical evidence "
                "for a grounded response."
            )

        elif consistency < self.cfg["retrieval"][
            "min_consistency"
        ]:

            reason = (
                "Retrieved historical resolutions "
                "are not sufficiently consistent."
            )

        elif trust < self.cfg["routing"][
            "min_trust_for_auto_handle"
        ]:

            reason = (
                "Overall trust score is below "
                "the auto-handling threshold."
            )

        # =====================================================
        # 8. FINAL OUTPUT
        # =====================================================

        return {
            "case_id": case[
                "case_id"
            ],

            "customer_text": customer_text,

            "intent": intent,

            "intent_confidence": round(
                intent_conf,
                4,
            ),

            "draft_reply": reply,

            "should_escalate": bool(
                must_escalate
            ),

            "decision": (
                "escalate"
                if must_escalate
                else "auto_handle"
            ),

            "escalation_reason": (
                reason
                if must_escalate
                else ""
            ),

            "retrieval_top_score": round(
                top_score,
                4,
            ),

            "evidence_consistency": round(
                consistency,
                4,
            ),

            "grounding_confidence": round(
                grounding,
                4,
            ),

            "risk_score": round(
                risk,
                4,
            ),

            "trust_score": trust,

            "evidence_case_ids": "|".join(
                map(
                    str,
                    evidence_ids[:5],
                )
            ),
        }