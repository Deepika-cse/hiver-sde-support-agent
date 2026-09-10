# \# Hiver SDE Intern — AI Customer Support Agent

# 

# An AI customer-support agent built for the Hiver SDE Intern take-home assignment using the \*\*Customer Support on Twitter (TWCS)\*\* dataset and \*\*AmazonHelp\*\* as the selected support brand.

# 

# The system:

# 

# 1\. reconstructs multi-turn customer-support cases,

# 2\. classifies the customer's intent,

# 3\. retrieves similar historical AmazonHelp interactions,

# 4\. generates a grounded support reply,

# 5\. evaluates evidence and risk,

# 6\. decides whether to `auto\_handle` or `escalate`,

# 7\. evaluates the system against a hand-labelled golden set,

# 8\. compares the approach with simple baselines.

# 

# \---

# 

# \## 1. System overview

# 

# ```text

# Incoming customer message

# &#x20;       ↓

# Conversation context

# &#x20;       ↓

# Intent classification

# &#x20;       ↓

# TF-IDF historical retrieval

# &#x20;       ↓

# Evidence consistency + risk checks

# &#x20;       ↓

# LLM grounded reply

# &#x20;       ↓

# Trust / routing checks

# &#x20;       ↓

# AUTO-HANDLE / ESCALATE + reason

# ```

# 

# The design intentionally separates \*\*intent classification\*\* from \*\*safe resolution\*\*. A case can have an obvious intent while still requiring account-specific or order-specific investigation.

# 

# \---

# 

# \## 2. Dataset

# 

# The project uses the Customer Support on Twitter (TWCS) dataset.

# 

# The downloaded dataset contains:

# 

# \* \*\*2,811,774 raw tweets\*\*

# \* Columns:

# 

# &#x20; \* `tweet\_id`

# &#x20; \* `author\_id`

# &#x20; \* `inbound`

# &#x20; \* `created\_at`

# &#x20; \* `text`

# &#x20; \* `response\_tweet\_id`

# &#x20; \* `in\_response\_to\_tweet\_id`

# 

# The selected support brand is \*\*AmazonHelp\*\*.

# 

# The loader reconstructs customer-support interactions into cases rather than treating every tweet independently.

# 

# This produced:

# 

# \* \*\*168,814 reconstructed AmazonHelp cases\*\*

# \* Context window: up to \*\*6 turns\*\*

# \* Mean context length: \*\*2.49 turns\*\*

# \* Median context length: \*\*1 turn\*\*

# 

# The full TWCS dataset is not committed to the repository.

# 

# Place the downloaded dataset at:

# 

# ```text

# data/raw/twcs.csv

# ```

# 

# \---

# 

# \## 3. Setup

# 

# Python 3.10+ is recommended.

# 

# \### Create virtual environment

# 

# Windows:

# 

# ```powershell

# python -m venv .venv

# .venv\\Scripts\\activate

# ```

# 

# macOS/Linux:

# 

# ```bash

# python -m venv .venv

# source .venv/bin/activate

# ```

# 

# Install dependencies:

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \---

# 

# \## 4. Environment variables

# 

# The current implementation uses:

# 

# ```text

# LLM\_PROVIDER=groq

# GROQ\_API\_KEY=your\_key\_here

# GROQ\_MODEL=openai/gpt-oss-20b

# ```

# 

# Do \*\*not\*\* commit `.env` or API keys.

# 

# A template is provided in:

# 

# ```text

# .env.example

# ```

# 

# The LLM provider can be changed through configuration, but the final reported experiment used:

# 

# ```text

# Provider: Groq

# Model: openai/gpt-oss-20b

# ```

# 

# \---

# 

# \## 5. First run without an API key

# 

# A small synthetic dataset is included so the pipeline structure can be tested without the full TWCS dataset or an API key.

# 

# ```bash

# python -m src.load\_data \\

# &#x20; --raw\_csv data/raw/sample\_raw.csv \\

# &#x20; --brand AmazonHelp \\

# &#x20; --out data/processed/threads.csv

# ```

# 

# Then:

# 

# ```bash

# python -m src.run\_pipeline \\

# &#x20; --threads\_csv data/processed/threads.csv \\

# &#x20; --n 10 \\

# &#x20; --out results/pipeline\_output.csv \\

# &#x20; --dry\_run

# ```

# 

# \---

# 

# \## 6. Reconstruct the real dataset

# 

# Place the TWCS file at:

# 

# ```text

# data/raw/twcs.csv

# ```

# 

# Discover candidate support accounts:

# 

# ```bash

# python -m src.discover\_brands \\

# &#x20; --raw\_csv data/raw/twcs.csv \\

# &#x20; --out results/brand\_candidates.csv \\

# &#x20; --top\_k 30

# ```

# 

# Then reconstruct the selected brand:

# 

# ```bash

# python -m src.load\_data \\

# &#x20; --raw\_csv data/raw/twcs.csv \\

# &#x20; --brand\_id YOUR\_SUPPORT\_AUTHOR\_ID \\

# &#x20; --brand\_name AmazonHelp \\

# &#x20; --out data/processed/threads.csv

# ```

# 

# The final experiment uses the reconstructed AmazonHelp cases in:

# 

# ```text

# data/processed/threads.csv

# ```

# 

# \---

# 

# \## 7. Intent taxonomy

# 

# The final taxonomy contains 10 intents:

# 

# | Intent                    | Description                                                              |

# | ------------------------- | ------------------------------------------------------------------------ |

# | `order\_issue`             | Problems with an order, missing/wrong items, or order-specific handling  |

# | `delivery\_issue`          | Delivery delays, failures, carrier/address problems, or delivery charges |

# | `package\_not\_received`    | Package marked/determined delivered but not received                     |

# | `return\_cancellation`     | Return or cancellation requests/problems                                 |

# | `refund\_payment`          | Refund, billing, payment, or charge problems                             |

# | `prime\_subscription`      | Prime membership, charges, or Prime benefits                             |

# | `account\_issue`           | Account access/state/management problems                                 |

# | `product\_technical\_issue` | Faulty products, technical failures, or troubleshooting                  |

# | `complaint\_followup`      | Repeated contact, unresolved interactions, or follow-up                  |

# | `other`                   | Cases outside the defined categories                                     |

# 

# The taxonomy is stored in:

# 

# ```text

# configs/taxonomy.yaml

# ```

# 

# \---

# 

# \## 8. Historical retrieval

# 

# The historical retriever uses \*\*TF-IDF\*\* with unigram and bigram features.

# 

# Why TF-IDF?

# 

# \* deterministic,

# \* transparent,

# \* inexpensive,

# \* easy to reproduce,

# \* requires no external embedding-model download.

# 

# The retriever searches historical AmazonHelp cases using:

# 

# ```text

# latest\_customer\_text + conversation\_context

# ```

# 

# The current case is excluded from its own retrieval results to avoid trivial self-retrieval leakage.

# 

# \---

# 

# \## 9. LLM response generation

# 

# The final experiment uses:

# 

# ```text

# Groq

# openai/gpt-oss-20b

# ```

# 

# The model receives the customer situation together with retrieved historical evidence and is instructed to use historical replies as \*\*evidence\*\*, not as authoritative instructions.

# 

# This is important because historical support responses can be incomplete, inconsistent, or outdated.

# 

# The generated output contains the predicted intent, evidence information, reply, trust/routing information, and escalation reason.

# 

# \---

# 

# \## 10. Routing and safety

# 

# The system does not rely only on LLM confidence.

# 

# Deterministic checks can force escalation for:

# 

# \* sensitive account situations,

# \* persistent/unresolved issues,

# \* insufficient information,

# \* payment/transaction problems,

# \* high-risk order situations,

# \* repeated follow-up requests,

# \* high-risk terms such as fraud, hacked accounts, unauthorized activity, chargebacks, legal threats, danger, or injury.

# 

# Configured thresholds include:

# 

# ```yaml

# routing:

# &#x20; min\_trust\_for\_auto\_handle: 0.62

# &#x20; max\_risk\_for\_auto\_handle: 0.45

# ```

# 

# Retrieval thresholds include:

# 

# ```yaml

# retrieval:

# &#x20; min\_top\_score: 0.18

# &#x20; strong\_top\_score: 0.45

# &#x20; min\_evidence\_items: 2

# &#x20; min\_consistency: 0.10

# ```

# 

# The design intentionally favors escalation when a response cannot be safely grounded.

# 

# \---

# 

# \## 11. Golden evaluation set

# 

# The final evaluation uses:

# 

# ```text

# data/golden/golden\_set.csv

# ```

# 

# It contains \*\*180 hand-labelled cases\*\*.

# 

# The golden set was created using diversity-aware sampling rather than simply taking the first 180 cases.

# 

# Human labels include:

# 

# \* `gold\_intent`

# \* `gold\_secondary\_intent`

# \* `gold\_should\_escalate`

# \* `gold\_escalation\_reason`

# \* `gold\_reply\_notes`

# 

# The labeling methodology is documented in:

# 

# ```text

# eval/LABELING\_GUIDE.md

# ```

# 

# The golden set is a human reference set and is not generated by the model.

# 

# \---

# 

# \## 12. Baselines

# 

# Three baselines were evaluated:

# 

# \### Majority baseline

# 

# Always predicts the most common intent.

# 

# \### Keyword baseline

# 

# Uses deterministic keyword/category matching.

# 

# \### TF-IDF + Logistic Regression

# 

# Uses TF-IDF features with a Logistic Regression classifier.

# 

# The baseline evaluation uses a stratified holdout from the labelled data.

# 

# \---

# 

# \## 13. Final evaluation results

# 

# The final proposed-agent evaluation covers all:

# 

# \*\*180 / 180 golden cases\*\*

# 

# | Metric                  |     Result |

# | ----------------------- | ---------: |

# | Intent accuracy         | \*\*55.00%\*\* |

# | Intent macro-F1         | \*\*56.11%\*\* |

# | Escalation precision    | \*\*60.61%\*\* |

# | Escalation recall       | \*\*86.96%\*\* |

# | Escalation F1           | \*\*71.43%\*\* |

# | False auto-handle rate  | \*\*13.04%\*\* |

# | False auto-handle count |     \*\*12\*\* |

# | Intent ECE              | \*\*0.1363\*\* |

# 

# The final predictions are stored at:

# 

# ```text

# results/groq\_predictions\_180\_final.csv

# ```

# 

# The final evaluation is stored under:

# 

# ```text

# results/eval\_final\_180/

# ```

# 

# \---

# 

# \## 14. Baseline comparison

# 

# The evaluated baseline results were:

# 

# | System                       | Intent Macro-F1 | Escalation Recall |

# | ---------------------------- | --------------: | ----------------: |

# | Majority                     |           4.00% |             0.00% |

# | Keyword                      |          21.64% |             0.00% |

# | TF-IDF + Logistic Regression |          18.28% |             0.00% |

# | \*\*Proposed agent\*\*           |      \*\*56.11%\*\* |        \*\*86.96%\*\* |

# 

# The simple baselines defaulted to non-escalation. Therefore, their zero escalation recall should not be interpreted as a meaningful safety advantage.

# 

# \---

# 

# \## 15. LLM judge

# 

# An LLM-judge evaluation pipeline is implemented in:

# 

# ```text

# eval/run\_judge.py

# ```

# 

# The intended workflow is:

# 

# ```text

# Generated reply

# &#x20;     ↓

# LLM judge

# &#x20;     ↓

# Reply-quality rubric

# &#x20;     ↓

# Human rating of same examples

# &#x20;     ↓

# Quadratic weighted Cohen's kappa

# ```

# 

# However, judge-human validation is \*\*not reported as completed\*\* for the final submission.

# 

# During execution, 8 LLM-judge cases were successfully completed before the Groq token/day limit was reached. The human rating form was not populated because an independent human-rated validation sample was not completed.

# 

# No judge score or judge-human agreement statistic is fabricated in this repository.

# 

# \---

# 

# \## 16. Failure analysis

# 

# The main observed failure modes were:

# 

# 1\. \*\*Ongoing issues hidden in conversation context\*\*

# 

# &#x20;  \* Example: `twcs\_273`

# &#x20;  \* The latest message alone under-represents the unresolved history.

# 

# 2\. \*\*Order-specific investigation\*\*

# 

# &#x20;  \* Example: `twcs\_663`

# &#x20;  \* Historical evidence cannot substitute for checking private order/account state.

# 

# 3\. \*\*Sparse delivery-investigation messages\*\*

# 

# &#x20;  \* Examples: `twcs\_2542`, `twcs\_2557`, `twcs\_3739`

# &#x20;  \* Intent can be clear while resolution evidence remains insufficient.

# 

# 4\. \*\*Multilingual/noisy text\*\*

# 

# &#x20;  \* Example: `twcs\_3739`

# &#x20;  \* Lexical retrieval is weaker across languages.

# 

# 5\. \*\*Multiple competing problem signals\*\*

# 

# &#x20;  \* Example: `twcs\_4828`

# &#x20;  \* Support-channel complaints can distract from the underlying product problem.

# 

# Detailed analysis is available in:

# 

# ```text

# reports/REPORT.md

# ```

# 

# \---

# 

# \## 17. What is misleading about the headline number?

# 

# The headline metric is \*\*86.96% escalation recall\*\*.

# 

# This is useful because missing an escalation-worthy case is an important safety failure.

# 

# However, it does not mean that 86.96% of cases can be safely resolved.

# 

# The system escalated \*\*132 of 180\*\* evaluated cases. Therefore, high escalation recall is partly achieved through conservative routing.

# 

# The final evaluation also contains only 180 hand-labelled cases, so the result may not represent the full production distribution.

# 

# The system still produced \*\*12 false auto-handles\*\*, which means some cases judged escalation-worthy by humans were routed toward automatic handling.

# 

# The system should therefore be viewed as a \*\*conservative support triage and drafting assistant\*\*, not as a fully autonomous support agent.

# 

# \---

# 

# \## 18. One-more-week plan

# 

# 1\. \*\*Separate intent confidence from resolution confidence.\*\*

# &#x20;  A model can confidently identify an intent while still lacking the account/order information needed to resolve the case safely.

# 

# 2\. \*\*Add multilingual semantic retrieval.\*\*

# &#x20;  Compare TF-IDF with multilingual embeddings, especially on non-English and noisy customer messages.

# 

# 3\. \*\*Complete independent judge-human validation.\*\*

# &#x20;  Have humans rate a fixed subset of generated replies and calculate agreement before using LLM-judge scores as a quality metric.

# 

# 4\. \*\*Evaluate routing by risk and intent.\*\*

# &#x20;  Measure false auto-handles separately for payment, account, delivery, technical, and other higher-risk categories.

# 

# The priority should be reducing unsafe auto-handling rather than maximizing a single aggregate accuracy number.

# 

# \---

# 

# \## 19. Repository structure

# 

# ```text

# .

# ├── configs/

# │   ├── taxonomy.yaml

# │   └── thresholds.yaml

# ├── data/

# │   ├── raw/

# │   ├── processed/

# │   └── golden/

# ├── decision\_log/

# │   └── DECISION\_LOG.md

# ├── eval/

# │   ├── LABELING\_GUIDE.md

# │   ├── HUMAN\_JUDGE\_FORM.csv

# │   ├── run\_baselines.py

# │   ├── run\_eval.py

# │   ├── run\_judge.py

# │   ├── judge\_agreement.py

# │   └── failure\_analysis.py

# ├── reports/

# │   └── REPORT.md

# ├── src/

# │   ├── agent/

# │   ├── retrieval/

# │   ├── baselines.py

# │   └── ...

# ├── tests/

# ├── .env.example

# ├── requirements.txt

# └── README.md

# ```

# 

# \---

# 

# \## 20. Tests

# 

# Run:

# 

# ```bash

# python -m pytest -q

# ```

# 

# Current result:

# 

# ```text

# 6 passed in 2.46s

# ```

# 

# \---

# 

# \## 21. Reproducibility

# 

# The final reported evaluation is based on:

# 

# \* the committed source code,

# \* the reconstructed AmazonHelp cases,

# \* the 180-case hand-labelled golden set,

# \* the configured TF-IDF retriever,

# \* Groq `openai/gpt-oss-20b`.

# 

# The full TWCS dataset is intentionally not committed because of dataset size/licensing considerations.

# 

# API keys must never be committed.

# 

# \---

# 

# \## 22. Key engineering takeaway

# 

# The main lesson from the evaluation is that \*\*classification confidence is not the same as resolution confidence\*\*.

# 

# A support agent may know that a customer has a delivery problem while still being unable to safely resolve it without access to order or carrier information.

# 

# The safest architecture is therefore:

# 

# ```text

# Classify the problem

# &#x20;       +

# Determine whether the problem is actually resolvable

# &#x20;       +

# Escalate when evidence or required private state is missing

# ```

# 

# This makes the system more appropriate for real support operations than optimizing only for intent accuracy.



