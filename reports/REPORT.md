# Hiver SDE Intern — AI Customer Support Agent

## 1. Executive Summary

**Selected brand:** AmazonHelp

**Headline result:** **86.96% escalation recall**

The system classifies incoming AmazonHelp customer cases into 10 support intents, retrieves similar historical support interactions using TF-IDF, generates a grounded response with an LLM, and applies deterministic safety/routing checks to decide whether to auto-handle or escalate.

On the 180-case hand-labelled golden set, the system achieved **55.00% intent accuracy**, **56.11% intent macro-F1**, and **86.96% escalation recall**, with a **13.04% false auto-handle rate**.

### Trust takeaway

The system is better suited to **assisted support and conservative triage** than unrestricted autonomous resolution. Its strongest property is identifying cases that should receive human attention, while intent classification and safe auto-handling still have meaningful failure modes.

## 2. Problem Framing

### Why this is difficult

- TWCS contains noisy, informal, multilingual and multi-turn customer conversations.
- Historical support resolutions may be incomplete or inconsistent.
- A fluent response is not necessarily a grounded or safe response.
- Auto-handling has asymmetric risk: an incorrect confident answer can be worse than escalating to a human.

### Data

| Item | Value |
|---|---:|
| Source | Customer Support on Twitter (TWCS) |
| Selected brand | AmazonHelp |
| Raw TWCS rows | 2,811,774 |
| Reconstructed AmazonHelp cases | 168,814 |
| Golden set | 180 hand-labelled cases |
| Maximum context | 6 turns |
| Mean context | 2.49 turns |
| Median context | 1 turn |

Cases were sampled using diversity-aware sampling rather than simply taking the first 180 rows.

The golden labels were manually assigned using `eval/LABELING_GUIDE.md`.

## 3. System Design

```text
Incoming customer message
        |
        v
Conversation context reconstruction
        |
        v
Intent classification
        |
        v
TF-IDF historical retrieval
        |
        v
Evidence consistency + risk checks
        |
        v
Grounded LLM response
        |
        v
Trust + routing decision
        |
        v
AUTO-HANDLE / ESCALATE
```

The design separates:

1. **Intent:** What is the customer's problem?
2. **Resolution confidence:** Can the problem be safely resolved with the evidence available to the system?

This is important because a message can have a clear intent while still requiring private order, account, payment, or carrier information.

## 4. Intent Taxonomy

| Intent | Definition | Count |
|---|---|---:|
| `order_issue` | Order problems, missing/wrong items, or order-specific handling | 12 |
| `delivery_issue` | Delivery delays, failures, carrier/address problems, or delivery charges | 44 |
| `package_not_received` | Package marked/determined delivered but not received | 11 |
| `return_cancellation` | Returns or cancellation requests/problems | 4 |
| `refund_payment` | Refunds, charges, billing, or payment problems | 12 |
| `prime_subscription` | Prime membership, charges, or benefits | 13 |
| `account_issue` | Account access, state, or management problems | 20 |
| `product_technical_issue` | Faulty products, technical failures, or troubleshooting | 23 |
| `complaint_followup` | Repeated contact, unresolved support, or follow-up requests | 19 |
| `other` | Cases outside the above categories | 22 |

## 5. Retrieval, Generation, and Routing

### Retrieval

Historical support examples are retrieved with TF-IDF over the latest customer text and conversation context using word unigrams and bigrams.

The current case is excluded from its own retrieval candidates to reduce trivial self-retrieval leakage.

### Generation

The LLM receives the customer message, conversation context, and retrieved historical support examples and generates a grounded customer-facing response.

The final implementation uses Groq:

```text
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-20b
```

### Routing

Routing combines model confidence, retrieval quality, evidence consistency, and deterministic safety checks.

Configured thresholds:

| Parameter | Value |
|---|---:|
| Minimum trust for auto-handling | 0.62 |
| Maximum risk for auto-handling | 0.45 |
| Minimum retrieval evidence items | 2 |
| Strong retrieval score | 0.45 |
| Minimum retrieval score | 0.18 |

High-risk signals include hacked/unauthorized accounts, fraud, chargebacks, legal threats, danger, injury, and similar language.

The system intentionally favors escalation when evidence is weak or the situation is unsafe to resolve automatically.

## 6. Evaluation Methodology

The final evaluation uses **180 hand-labelled cases**, within the required 150–250 case range.

### Golden-set distribution

| Intent | Cases |
|---|---:|
| delivery_issue | 44 |
| product_technical_issue | 23 |
| other | 22 |
| account_issue | 20 |
| complaint_followup | 19 |
| prime_subscription | 13 |
| order_issue | 12 |
| refund_payment | 12 |
| package_not_received | 11 |
| return_cancellation | 4 |
| **Total** | **180** |

Escalation labels:

- Gold escalation: 92
- Gold non-escalation: 88

The retriever excludes the current case from candidate evidence to reduce leakage.

## 7. Baselines

Three simple baselines were evaluated on a stratified holdout of 36 cases.

| System | Accuracy | Macro-F1 | Escalation Recall |
|---|---:|---:|---:|
| Majority | 25.00% | 4.00% | 0.00% |
| Keyword | 27.78% | 21.64% | 0.00% |
| TF-IDF + Logistic Regression | 19.44% | 18.28% | 0.00% |

The baselines defaulted to non-escalation. Their zero escalation recall is therefore not evidence of safety.

## 8. Final Agent Results

| Metric | Result |
|---|---:|
| Intent accuracy | **55.00%** |
| Intent macro-F1 | **56.11%** |
| Escalation precision | **60.61%** |
| Escalation recall | **86.96%** |
| Escalation F1 | **71.43%** |
| False auto-handles | **12 / 92** |
| False auto-handle rate | **13.04%** |
| Intent ECE | **0.136** |

The system escalated 132 of 180 evaluated cases and auto-handled 48.

The headline strength is high escalation recall, but the 12 false auto-handles demonstrate that conservative routing is not yet sufficient for unrestricted autonomous support.

## 9. LLM Judge and Human Validation

A separate LLM judge evaluated 30 generated responses using five dimensions:

- groundedness
- helpfulness
- correctness
- tone
- overall quality

### LLM-judge descriptive results

| Dimension | Mean |
|---|---:|
| Groundedness | 3.70 / 5 |
| Helpfulness | 3.33 / 5 |
| Correctness | 4.80 / 5 |
| Tone | 4.23 / 5 |
| Overall | 3.70 / 5 |

### Human validation

A manually rated 10-case validation subset was compared with the LLM-judge ratings using quadratic weighted Cohen's kappa.

| Dimension | Kappa |
|---|---:|
| Groundedness | 0.123 |
| Helpfulness | 0.565 |
| Correctness | 0.370 |
| Tone | 0.556 |
| Overall | 0.556 |

The agreement varies by dimension. Groundedness agreement is weak, while helpfulness, tone, and overall quality show moderate agreement on this small sample.

Because the human validation sample contains only 10 cases, these values should be treated as diagnostic rather than definitive evidence of judge reliability.

Artifacts:

- `results/judge_30_retry.csv`
- `eval/HUMAN_JUDGE_FORM.csv`
- `results/judge_agreement_10.csv`

## 10. Top Five Failure Modes

### Failure 1 — Ongoing issue hidden in conversation context

**Examples:** `twcs_273`, `twcs_3752`, `twcs_3754`, `twcs_3755`

`twcs_273` describes a Fire TV Stick issue that remained unresolved after phone support and mentions that the warranty had expired.

**Observed behavior:** The intent was correctly classified as a product technical issue, but routing could still be too permissive.

**Hypothesis:** The strongest escalation signal can be distributed across conversation history instead of appearing in the latest message.

**Fix:** Add an explicit persistence/context-summary feature to routing.

### Failure 2 — Order-specific investigation

**Examples:** `twcs_663`, `twcs_1723`, `twcs_1725`, `twcs_3722`

`twcs_663` describes a delivery-charge problem requiring order-specific investigation.

**Observed behavior:** Historical similarity can make the case appear answerable.

**Hypothesis:** Retrieval cannot establish private customer-specific order state.

**Fix:** Add a `requires_account_or_order_lookup` signal that forces escalation.

### Failure 3 — Sparse delivery-investigation language

**Examples:** `twcs_2542`, `twcs_2557`, `twcs_3739`

`twcs_3739` reports that the carrier could not locate the customer's address.

**Observed behavior:** The system can identify the delivery intent without having enough information for safe resolution.

**Hypothesis:** Intent confidence can be high while resolution confidence is low.

**Fix:** Explicitly separate intent confidence from resolution confidence.

### Failure 4 — Multilingual and noisy customer text

**Example:** `twcs_3739`

The Portuguese delivery message differs substantially from the dominant English-language examples.

**Hypothesis:** TF-IDF is less robust to cross-language semantic similarity.

**Fix:** Evaluate multilingual semantic retrieval and report performance by language.

### Failure 5 — Faulty product plus support-access problem

**Example:** `twcs_4828`

The model predicted `order_issue`, while the human label was `product_technical_issue`.

**Hypothesis:** Support-channel language can compete with the underlying customer problem.

**Fix:** Use secondary-intent reasoning and prioritize the underlying issue.

## 11. What Is Misleading About the Headline Number?

The headline **86.96% escalation recall** is useful, but it can be misleading if interpreted as evidence that the system is ready for autonomous support.

First, escalation recall is achieved partly through conservative routing. The system escalates **132 of 180** evaluated cases, leaving 48 for auto-handling.

Second, the golden set contains only 180 cases and is diversity-oriented. Its distribution may differ from real incoming AmazonHelp traffic.

Third, escalation recall does not directly measure whether generated replies are correct or well grounded.

Finally, the system produced **12 false auto-handles**, meaning some cases that humans considered escalation-worthy were routed to auto-handling.

**More honest takeaway:** the system is promising as a **conservative support triage and drafting assistant**, but the current evidence is insufficient to justify broad autonomous customer handling.

## 12. What I Would Do With One More Week

1. **Separate classification confidence from resolution confidence.**
   A clear intent does not mean the case can be resolved without private customer information.

2. **Add multilingual semantic retrieval.**
   Compare multilingual embeddings against TF-IDF, especially on non-English and noisy messages.

3. **Expand judge-human validation.**
   Increase the human-rated sample and repeat the agreement analysis.

4. **Evaluate routing by intent and risk tier.**
   Report false auto-handles separately for account, payment, delivery, technical, and other high-risk categories.

The priority should be reducing **false auto-handles**, not maximizing raw intent accuracy.

## 13. Decision Log

1. **Conversation reconstruction:** Preserve multi-turn support context.
2. **AmazonHelp selection:** Broad coverage of support scenarios.
3. **Fixed taxonomy:** Reduce evaluation drift.
4. **Ten intents:** Balance coverage and annotation consistency.
5. **TF-IDF retrieval:** Lightweight, interpretable, reproducible baseline.
6. **Self-retrieval exclusion:** Reduce trivial leakage.
7. **Separate retrieval and generation:** Historical examples provide evidence; the LLM drafts the response.
8. **Intent vs. resolution confidence:** Understanding the problem is not equivalent to resolving it.
9. **Conservative escalation:** Human review is safer when evidence is insufficient.
10. **Deterministic risk checks:** High-risk cases should not depend solely on model judgment.
11. **False auto-handle metric:** Capture unsafe routing that aggregate recall can hide.
12. **Simple baselines:** Establish a meaningful performance reference.
13. **Separate LLM judge:** Evaluate response quality independently from generation.
14. **Human judge validation:** Test whether judge scores align with human ratings.
15. **Diagnostic kappa:** Avoid overstating reliability from a 10-case sample.

## 14. Reproducibility

### Baselines

```powershell
python -m eval.run_baselines
```

### Final evaluation

```powershell
python -m eval.run_eval `
  --predictions results/groq_predictions_180_final.csv `
  --gold data/golden/golden_set.csv `
  --out_dir results\eval_final_180
```

### LLM judge

```powershell
python -m eval.run_judge `
  --predictions results/groq_predictions_180_final.csv `
  --threads data/processed/threads.csv `
  --n 30 `
  --out results/judge_30_retry.csv
```

### Judge-human agreement

```powershell
python -m eval.judge_agreement `
  --judge results/judge_30_retry.csv `
  --human eval\HUMAN_JUDGE_FORM.csv `
  --out results\judge_agreement_10.csv
```

## 15. Conclusion

The proposed AmazonHelp support agent substantially outperforms the simple baselines on intent classification while achieving high escalation recall.

Its main limitation is that historical similarity and intent confidence do not guarantee that a case can actually be resolved without account-specific investigation.

The system is therefore best positioned as a **conservative triage and drafting assistant today**, with safer autonomous handling as the next engineering target.
