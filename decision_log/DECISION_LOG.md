# Decision Log

1. **Brand selection** — chose **AmazonHelp** because the reconstructed dataset provided a large number of support cases and enough diversity to define a meaningful support taxonomy.

2. **Case definition** — used customer→brand resolution units instead of individual tweets because support decisions require conversation context. This produced **168,814 reconstructed cases** from **2,811,774 raw TWCS tweets**.

3. **Context window** — retained up to **6 turns** per reconstructed case. The resulting cases have a mean of **2.49 turns**, with a median of 1 and a maximum of 6. This keeps relevant history while limiting unrelated conversation noise.

4. **Intent count** — froze **10 intents** after inspecting AmazonHelp examples and manually consolidating overlapping customer problems. The taxonomy covers order, delivery, package-not-received, returns/cancellation, refunds/payments, Prime, account, technical, follow-up, and other cases.

5. **Golden-set size** — used **180 hand-labelled examples**, satisfying the required 150–250 range while providing coverage across all 10 intents.

6. **Golden sampling** — used diversity-aware sampling rather than taking the first 180 rows to reduce ordering bias and improve coverage of different support situations.

7. **Retrieval method** — used **TF-IDF with unigram/bigram features** as the historical retriever because it is reproducible, transparent, computationally inexpensive, and requires no external embedding-model download.

8. **Evidence exclusion** — excluded the current case from its own retrieval results to prevent trivial self-retrieval and reduce contamination of the evidence set.

9. **LLM call design** — used one structured LLM call per case to keep inference complexity manageable while allowing the model to combine the customer message, conversation context, and retrieved historical evidence.

10. **Grounding rule** — instructed the model to treat historical brand replies as **evidence rather than authoritative instructions**. This reduces the risk of blindly copying an outdated historical resolution.

11. **Risk guardrails** — deterministic high-risk signals can force escalation before trusting model fluency. Examples include hacked or unauthorized accounts, fraud, chargebacks, legal threats, danger, injury, and similar high-risk language.

12. **Conflict handling** — retrieval consistency is incorporated into trust/routing. When historical resolutions are weak or inconsistent, the system becomes more conservative instead of selecting a fluent answer solely because it sounds plausible.

13. **Routing threshold** — used an explicit auto-handle trust threshold of **0.62** and a maximum risk threshold of **0.45**. The final evaluation was not tuned against individual test examples; routing also includes deterministic safety rules intended to prevent unsafe auto-handling.

14. **Judge validation** — implemented an LLM-judge workflow and a separate human-rating form, but did **not** claim judge-human agreement because the judge provider reached its token/day limit before the human validation set was completed. No agreement statistic is fabricated.

15. **Headline metric** — selected **escalation recall** as the headline because missing an escalation-worthy customer case is more safety-critical than merely producing a high aggregate intent score. The report explicitly discloses that this metric can look strong because the system is conservative and escalates many cases.
