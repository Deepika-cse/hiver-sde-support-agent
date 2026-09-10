# Hiver SDE Intern — AI Customer Support Agent

AI customer-support agent built for the Hiver SDE Intern take-home assignment using the Customer Support on Twitter (TWCS) dataset and AmazonHelp as the selected brand.

## 1. Executive Summary

The agent:

- classifies incoming customer cases into 10 support intents
- retrieves similar historical brand resolutions
- drafts a grounded support response with an LLM
- decides whether to `auto_handle` or `escalate`
- provides an escalation reason
- evaluates performance on a 180-case hand-labelled golden set
- compares against simple baselines
- evaluates response quality with an LLM judge and human validation
- analyzes failure modes and false auto-handling

**Core design principle:** intent confidence is not the same as resolution confidence.

A customer message can have a clear intent while still requiring private order, account, payment, or carrier information that the system does not have.

## 2. System Overview

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
Grounded LLM reply generation
        |
        v
Trust + routing decision
        |
        v
AUTO-HANDLE / ESCALATE
```

The system separates two questions:

1. What is the customer's problem?
2. Can it be safely resolved with the available evidence?

This distinction is important for support automation because historical similarity does not provide access to private customer state.

## 3. Dataset

**Source:** Customer Support on Twitter (TWCS)

**Selected brand:** AmazonHelp

| Dataset stage | Count |
|---|---:|
| Raw TWCS rows | 2,811,774 |
| Reconstructed AmazonHelp cases | 168,814 |
| Hand-labelled golden cases | 180 |

Conversation reconstruction produces a support-case unit rather than treating every tweet independently.

- Maximum context: 6 turns
- Mean context length: 2.49 turns
- Median context length: 1 turn
- Golden-set sampling: diversity-aware rather than first-N sampling

The golden labels were manually assigned using `eval/LABELING_GUIDE.md`.

## 4. Intent Taxonomy

| Intent | Definition | Golden count |
|---|---|---:|
| `order_issue` | Order problems, missing/wrong items, or order-specific handling | 12 |
| `delivery_issue` | Delivery delays, failures, carrier/address problems, or delivery charges | 44 |
| `package_not_received` | Package marked/determined delivered but not received | 11 |
| `return_cancellation` | Returns or order-cancellation requests/problems | 4 |
| `refund_payment` | Refunds, charges, billing, or payment problems | 12 |
| `prime_subscription` | Prime membership, charges, or benefits | 13 |
| `account_issue` | Account access, account state, or account management | 20 |
| `product_technical_issue` | Faulty products, technical failures, or troubleshooting | 23 |
| `complaint_followup` | Repeated contact, unresolved support, or follow-up requests | 19 |
| `other` | Cases outside the above categories | 22 |

## 5. Retrieval and Generation

### Retrieval

Historical support examples are retrieved using TF-IDF over:

- latest customer text
- conversation context
- word unigrams and bigrams

The current case is excluded from its own retrieval candidates to reduce trivial self-retrieval leakage.

### Generation

The LLM receives the customer message, conversation context, and retrieved historical support examples and is instructed to produce a grounded response.

The final implementation uses Groq with:

```text
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-20b
```

The generated response is evaluated separately from the retrieval and routing components.

## 6. Safety and Routing

Routing combines:

- model confidence
- retrieval quality
- evidence consistency
- risk signals
- deterministic safety checks

Configured thresholds:

| Parameter | Value |
|---|---:|
| Minimum trust for auto-handling | 0.62 |
| Maximum risk for auto-handling | 0.45 |
| Minimum retrieval evidence items | 2 |
| Strong retrieval score | 0.45 |
| Minimum retrieval score | 0.18 |

High-risk signals include language involving hacked or unauthorized accounts, fraud, chargebacks, legal threats, danger, injury, and similar situations.

Additional deterministic checks cover sensitive account issues, persistent or unresolved problems, insufficient information, payment/transaction issues, high-risk order situations, and follow-up requests.

The routing policy intentionally favors escalation when the available evidence is insufficient for safe autonomous resolution.

## 7. Golden Evaluation

The final evaluation uses **180 hand-labelled cases**, satisfying the required 150–250 case range.

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

## 8. Baselines

Three simple baselines were evaluated on a stratified holdout:

1. Majority-class baseline
2. Keyword baseline
3. TF-IDF + Logistic Regression

The holdout contained 36 cases, with 144 cases used for training where applicable.

| System | Accuracy | Macro-F1 | Escalation Recall |
|---|---:|---:|---:|
| Majority | 25.00% | 4.00% | 0.00% |
| Keyword | 27.78% | 21.64% | 0.00% |
| TF-IDF + Logistic Regression | 19.44% | 18.28% | 0.00% |

The simple baselines defaulted to non-escalation. Their zero escalation recall therefore does not demonstrate safety.

## 9. Final Agent Results

Final evaluation on all 180 golden cases:

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

The agent escalated 132 of 180 evaluated cases and auto-handled 48.

The strongest result is high escalation recall. However, the 12 false auto-handles show that conservative routing still has important failure cases.

## 10. LLM Judge Evaluation

A separate LLM judge evaluated 30 generated responses using five dimensions:

- groundedness
- helpfulness
- correctness
- tone
- overall quality

### LLM judge means

| Dimension | Mean |
|---|---:|
| Groundedness | 3.70 / 5 |
| Helpfulness | 3.33 / 5 |
| Correctness | 4.80 / 5 |
| Tone | 4.23 / 5 |
| Overall | 3.70 / 5 |

### Human validation

A manually rated 10-case validation subset was compared with the LLM judge using quadratic weighted Cohen's kappa.

| Dimension | Kappa |
|---|---:|
| Groundedness | 0.123 |
| Helpfulness | 0.565 |
| Correctness | 0.370 |
| Tone | 0.556 |
| Overall | 0.556 |

The agreement varies by dimension. Groundedness agreement is weak, while helpfulness, tone, and overall quality show moderate agreement on this small sample.

Because only 10 cases were human-rated, these statistics should be treated as diagnostic rather than definitive evidence of judge reliability.

Evaluation artifacts:

- `results/judge_30_retry.csv`
- `eval/HUMAN_JUDGE_FORM.csv`
- `results/judge_agreement_10.csv`

## 11. Top Failure Modes

### 1. Ongoing issue hidden in conversation context

**Examples:** `twcs_273`, `twcs_3752`, `twcs_3754`, `twcs_3755`

A customer can have an unresolved issue across multiple turns even when the latest message looks routine.

**Observed problem:** The intent may be classified correctly while routing is too permissive.

**Hypothesis:** Over-weighting the latest customer message can hide persistence and previous troubleshooting.

**Next fix:** Add an explicit persistence/context-summary feature to routing.

### 2. Order-specific investigation

**Examples:** `twcs_663`, `twcs_1723`, `twcs_1725`, `twcs_3722`

Some cases require checking private order or account state.

**Observed problem:** Historical similarity can make these cases appear answerable.

**Hypothesis:** Retrieval cannot establish customer-specific facts.

**Next fix:** Add a `requires_account_or_order_lookup` signal that forces escalation.

### 3. Sparse delivery-investigation language

**Examples:** `twcs_2542`, `twcs_2557`, `twcs_3739`

Short delivery messages can have little evidence even when the intent is clear.

**Hypothesis:** Intent confidence can be high while resolution confidence is low.

**Next fix:** Explicitly separate intent confidence from resolution confidence.

### 4. Multilingual and noisy text

**Example:** `twcs_3739`

Lexical retrieval is less robust when the query language differs from the dominant historical corpus.

**Next fix:** Evaluate multilingual semantic retrieval and report performance by language.

### 5. Faulty product plus support-access problem

**Example:** `twcs_4828`

The model predicted `order_issue` while the human label was `product_technical_issue`.

**Hypothesis:** Support-channel language can compete with the underlying customer problem.

**Next fix:** Use secondary-intent reasoning and prioritize the underlying issue.

## 12. What Is Misleading About the Headline Number?

The headline **86.96% escalation recall** should not be interpreted as evidence that the system is ready for autonomous customer support.

Three limitations matter:

1. **High recall comes with high escalation volume.**
   The system escalated 132 of 180 cases.

2. **The golden set is small and diversity-oriented.**
   Its distribution may differ from real incoming AmazonHelp traffic.

3. **Escalation recall does not measure reply quality.**
   It says whether escalation-worthy cases were detected, not whether generated responses were correct or grounded.

The system still produced **12 false auto-handles**.

The more honest conclusion is that the system is promising as a **conservative support triage and drafting assistant**, not as a broadly autonomous support agent.

## 13. One-Week Improvement Plan

### Priority 1 — Separate intent and resolution confidence

A clear intent should not automatically imply that the case can be resolved without private customer information.

### Priority 2 — Improve multilingual retrieval

Test multilingual embeddings against TF-IDF, especially on non-English and noisy messages.

### Priority 3 — Expand judge-human validation

Increase the human-rated sample and repeat the agreement analysis before relying heavily on LLM-judge scores.

### Priority 4 — Evaluate routing by risk tier

Report false auto-handles separately for account, payment, delivery, technical, and other high-risk categories.

The main optimization target should be **reducing false auto-handles**, not maximizing raw intent accuracy.

## 14. Decision Log

1. **Reconstruct conversations:** Support interactions are conversational, so context is retained.
2. **Select AmazonHelp:** Provides a broad range of customer-support scenarios.
3. **Fix taxonomy before evaluation:** Reduces evaluation drift.
4. **Use 10 explicit intents:** Balances coverage with annotation consistency.
5. **Start with TF-IDF:** Lightweight, interpretable, reproducible retrieval baseline.
6. **Exclude self-retrieval:** Prevents trivial leakage from the current case.
7. **Separate retrieval and generation:** Retrieved examples provide evidence; the LLM drafts the response.
8. **Separate intent from resolution confidence:** Understanding a problem is not equivalent to being able to resolve it.
9. **Prefer escalation under uncertainty:** Incorrect autonomous support can be more costly than human review.
10. **Use deterministic risk checks:** Certain high-risk situations should not depend solely on model judgment.
11. **Measure false auto-handles:** Aggregate escalation recall alone can hide unsafe routing.
12. **Include simple baselines:** Establishes whether the proposed system adds value.
13. **Use a separate LLM judge:** Keeps response-quality evaluation separate from generation.
14. **Validate the judge with humans:** Avoids assuming LLM-judge scores are automatically reliable.
15. **Treat kappa as diagnostic:** Ten human-rated cases are insufficient for definitive reliability claims.

## 15. Reproducibility

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

## 16. Project Structure

```text
hiver-sde-assignment/
├── configs/
│   ├── taxonomy.yaml
│   └── thresholds.yaml
├── data/
│   ├── golden/
│   │   └── golden_set.csv
│   └── processed/
│       └── threads.csv
├── eval/
│   ├── LABELING_GUIDE.md
│   ├── HUMAN_JUDGE_FORM.csv
│   ├── judge_agreement.py
│   ├── run_baselines.py
│   ├── run_eval.py
│   └── run_judge.py
├── reports/
│   └── REPORT.md
├── results/
│   ├── groq_predictions_180_final.csv
│   ├── judge_30_retry.csv
│   └── judge_agreement_10.csv
└── src/
    ├── baselines.py
    ├── config.py
    └── retrieval/
        └── tfidf.py
```

## 17. Final Positioning

This project demonstrates an end-to-end support-agent pipeline covering:

**conversation reconstruction → intent classification → historical retrieval → grounded generation → safety/routing → evaluation → failure analysis**

The evaluation supports a conservative deployment position:

> **Use the system to assist support agents and triage incoming cases, while keeping human review for uncertain or high-risk situations.**

The next engineering step is not simply a larger model. It is better estimation of whether the system has enough evidence and access to safely resolve a specific customer case.
