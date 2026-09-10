# \# Hiver SDE Intern — AI Customer Support Agent

# 

# \## 1. Executive summary

# 

# \*\*Selected brand:\*\* AmazonHelp

# 

# \*\*Headline result:\*\* \*\*86.96% escalation recall\*\*

# 

# The system classifies incoming AmazonHelp customer cases into 10 support intents, retrieves similar historical support interactions using TF-IDF, generates a grounded response with an LLM, and applies deterministic safety/routing checks to decide whether to auto-handle or escalate.

# 

# On the 180-case hand-labelled golden set, the system achieved \*\*55.00% intent accuracy\*\*, \*\*56.11% intent macro-F1\*\*, and \*\*86.96% escalation recall\*\*, with a \*\*13.04% false auto-handle rate\*\*.

# 

# \### Trust takeaway

# 

# The system is better suited to \*\*assisted support and conservative triage\*\* than unrestricted autonomous resolution. Its strongest property is identifying cases that should receive human attention, while intent classification and safe auto-handling still have meaningful failure modes.

# 

# \---

# 

# \## 2. Problem framing

# 

# \### Why this is difficult

# 

# \* TWCS contains noisy, informal, multilingual and multi-turn customer conversations.

# \* Historical support resolutions may be incomplete or inconsistent.

# \* A fluent response is not necessarily a grounded or safe response.

# \* Auto-handling has asymmetric risk: an incorrect confident answer can be worse than escalating to a human.

# 

# \### Data

# 

# \* \*\*Source:\*\* Customer Support on Twitter (TWCS)

# \* \*\*Selected brand:\*\* AmazonHelp

# \* \*\*Raw TWCS rows:\*\* 2,811,774 tweets

# \* \*\*Reconstructed AmazonHelp cases:\*\* 168,814

# \* \*\*Golden set:\*\* 180 hand-labelled cases

# \* \*\*Context window:\*\* up to 6 turns; mean 2.49 turns per reconstructed case

# \* \*\*Sampling:\*\* diversity-aware sampling rather than taking the first N cases

# \* \*\*Golden labels:\*\* manually assigned intent and escalation decisions using `eval/LABELING\_GUIDE.md`

# 

# The reconstructed case unit is a customer-support interaction rather than an individual tweet. This preserves conversation context needed for intent and escalation decisions.

# 

# \---

# 

# \## 3. System

# 

# ```text

# Incoming customer message

# &#x20;       ↓

# Conversation context

# &#x20;       ↓

# Intent classifier / LLM

# &#x20;       ↓

# TF-IDF historical retrieval

# &#x20;       ↓

# Evidence consistency + risk checks

# &#x20;       ↓

# Grounded LLM draft

# &#x20;       ↓

# Trust / routing checks

# &#x20;       ↓

# AUTO-HANDLE / ESCALATE + reason

# ```

# 

# \### Intent taxonomy

# 

# | Intent                    | Definition                                                                           | Golden count |

# | ------------------------- | ------------------------------------------------------------------------------------ | -----------: |

# | `order\_issue`             | Problems with an order, missing/wrong items, or order-specific handling              |           12 |

# | `delivery\_issue`          | Delays, delivery failures, carrier/address problems, or delivery charges             |           44 |

# | `package\_not\_received`    | Customer reports that a package was marked/determined delivered but was not received |           11 |

# | `return\_cancellation`     | Requests or problems involving returns or cancelling an order                        |            4 |

# | `refund\_payment`          | Refunds, charges, billing or payment-related problems                                |           12 |

# | `prime\_subscription`      | Prime membership, Prime charges, or Prime-related benefits                           |           13 |

# | `account\_issue`           | Account access, account state, or account-management problems                        |           20 |

# | `product\_technical\_issue` | Faulty products, technical failures, or troubleshooting issues                       |           23 |

# | `complaint\_followup`      | Repeated contact, unresolved support interactions, or follow-up requests             |           19 |

# | `other`                   | Cases that do not fit the above support categories                                   |           22 |

# 

# \### Routing policy

# 

# The routing layer combines model confidence, retrieved evidence, consistency, and deterministic safety checks.

# 

# Configured thresholds include:

# 

# \* minimum trust for auto-handling: \*\*0.62\*\*

# \* maximum risk score for auto-handling: \*\*0.45\*\*

# \* minimum retrieval evidence items: \*\*2\*\*

# \* strong retrieval score: \*\*0.45\*\*

# \* minimum retrieval score: \*\*0.18\*\*

# 

# High-risk signals such as hacked/unauthorized accounts, fraud, chargebacks, legal threats, danger, injury, or similar language can force escalation.

# 

# Additional deterministic checks cover sensitive account issues, persistent/unresolved problems, insufficient information, payment/transaction issues, high-risk order situations, follow-up requests, and specific review signals.

# 

# The design intentionally biases toward escalation when evidence is weak or the customer situation appears unsafe to resolve automatically.

# 

# \---

# 

# \## 4. Evaluation

# 

# \### Results

# 

# | System                       | Intent Macro-F1 | Escalation Recall | False Auto-Handle Rate |

# | ---------------------------- | --------------: | ----------------: | ---------------------: |

# | Majority baseline            |           4.00% |             0.00% |                 0.00%\* |

# | Keyword baseline             |          21.64% |             0.00% |                 0.00%\* |

# | TF-IDF + Logistic Regression |          18.28% |             0.00% |                 0.00%\* |

# | \*\*Proposed agent\*\*           |      \*\*56.11%\*\* |        \*\*86.96%\*\* |             \*\*13.04%\*\* |

# 

# \*The simple baselines defaulted to non-escalation, so their apparent zero false-auto metric is not evidence of safety; they simply never escalated. Their auto-handle rate was effectively 100%.

# 

# The proposed agent therefore provides a substantially stronger intent signal than the trivial/simple baselines while also identifying most gold escalation cases.

# 

# Additional proposed-agent results:

# 

# \* Intent accuracy: \*\*55.00%\*\*

# \* Escalation precision: \*\*60.61%\*\*

# \* Escalation F1: \*\*71.43%\*\*

# \* False auto-handles: \*\*12 / 92 gold escalation cases\*\*

# \* Intent ECE: \*\*0.136\*\*

# 

# \### Golden-set methodology

# 

# The golden set contains \*\*180 hand-labelled cases\*\*, within the required 150–250 range.

# 

# Cases were sampled from reconstructed AmazonHelp support interactions using diversity-aware sampling rather than simply selecting the first 180 rows. The intent taxonomy was fixed before final evaluation. Human labels include primary intent, optional secondary intent, escalation decision, escalation reason, and reply notes.

# 

# The current golden set contains:

# 

# \* 44 delivery issues

# \* 23 product technical issues

# \* 22 other

# \* 20 account issues

# \* 19 complaint follow-ups

# \* 13 Prime subscription cases

# \* 12 order issues

# \* 12 refund/payment cases

# \* 11 package-not-received cases

# \* 4 return/cancellation cases

# 

# The retriever explicitly excludes the current case from its own evidence results, reducing trivial self-retrieval leakage.

# 

# \### LLM judge validation

# 

# The LLM judge was implemented as a separate evaluation component using the same generated predictions.

# 

# However, \*\*judge-human validation is not reported as completed\*\*. Eight LLM-judge cases were successfully evaluated before the Groq token/day limit was reached, but the human rating form was not populated with an independent human-rated subset.

# 

# Therefore:

# 

# \* judge-human sample: \*\*0 completed\*\*

# \* quadratic weighted Cohen's kappa: \*\*not available\*\*

# \* judge-human disagreements: \*\*not available\*\*

# 

# This is intentionally left unreported rather than fabricated.

# 

# \---

# 

# \## 5. Failure analysis — top 5

# 

# \### Failure 1 — Ongoing issue hidden in conversation context

# 

# \*\*Real examples:\*\* `twcs\_273`, `twcs\_3752`, `twcs\_3754`, `twcs\_3755`

# 

# For example, `twcs\_273` describes a Fire TV Stick issue that remained unresolved after phone support and mentions that the warranty had expired.

# 

# \*\*Observed behavior:\*\* The model correctly classified the intent as a product technical issue but sometimes treated the latest message as sufficiently routine for auto-handling.

# 

# \*\*Why it failed:\*\* The strongest escalation signal was distributed across the conversation history rather than contained in a single explicit phrase.

# 

# \*\*Hypothesis:\*\* A classifier that gives too much weight to the latest customer message can miss persistence and prior troubleshooting.

# 

# \*\*Fix:\*\* Add a dedicated context summarization/persistence feature and explicitly pass unresolved prior attempts into the routing decision.

# 

# \---

# 

# \### Failure 2 — Order-specific investigation

# 

# \*\*Real examples:\*\* `twcs\_663`, `twcs\_1723`, `twcs\_1725`, `twcs\_3722`

# 

# `twcs\_663`, for example, says the customer paid for delivery that did not arrive and requests a delivery-charge refund.

# 

# \*\*Observed behavior:\*\* The intent was often correct, but the system sometimes considered the case safe to auto-handle.

# 

# \*\*Why it failed:\*\* These cases require checking account/order-specific information that historical retrieval cannot establish.

# 

# \*\*Hypothesis:\*\* Historical similarity can make a case look answerable even when the actual resolution depends on private order state.

# 

# \*\*Fix:\*\* Introduce an explicit "requires account/order lookup" signal that forces escalation when resolution depends on unavailable customer-specific state.

# 

# \---

# 

# \### Failure 3 — Sparse delivery-investigation language

# 

# \*\*Real examples:\*\* `twcs\_2542`, `twcs\_2557`, `twcs\_3739`

# 

# `twcs\_3739` reports that the carrier could not locate the customer's address.

# 

# \*\*Observed behavior:\*\* The system classified these as delivery issues but did not always escalate.

# 

# \*\*Why it failed:\*\* The messages contain little descriptive context, while the underlying issue requires carrier/order investigation.

# 

# \*\*Hypothesis:\*\* Sparse messages can receive artificially high confidence from a broad intent category without providing enough evidence for a safe response.

# 

# \*\*Fix:\*\* Separate \*\*intent confidence\*\* from \*\*resolution confidence\*\*. A confident delivery classification should not imply that the case is safe to resolve automatically.

# 

# \---

# 

# \### Failure 4 — Multilingual and noisy customer text

# 

# \*\*Real example:\*\* `twcs\_3739`

# 

# The Portuguese delivery message contains a clear delivery problem but differs substantially from the dominant English-language examples.

# 

# \*\*Observed behavior:\*\* The system identified the general delivery intent but routing confidence was not sufficiently conservative.

# 

# \*\*Why it failed:\*\* TF-IDF retrieval and lexical features are less robust when the query language differs from much of the historical corpus.

# 

# \*\*Hypothesis:\*\* Lexical retrieval loses semantic similarity across languages.

# 

# \*\*Fix:\*\* Add multilingual semantic retrieval or a lightweight multilingual embedding model and evaluate performance separately by language.

# 

# \---

# 

# \### Failure 5 — Faulty product + support-access problem

# 

# \*\*Real example:\*\* `twcs\_4828`

# 

# The customer says the website is impossible to navigate and that they are trying to contact Amazon about a faulty item.

# 

# \*\*Observed behavior:\*\* The model predicted `order\_issue`, while the human label was `product\_technical\_issue`, and the case should have received human attention.

# 

# \*\*Why it failed:\*\* Two signals compete: difficulty navigating support and an underlying faulty product.

# 

# \*\*Hypothesis:\*\* The model over-weighted the support/order workflow language instead of the actual customer problem.

# 

# \*\*Fix:\*\* Use secondary-intent reasoning and explicitly prioritize the underlying customer problem over the support-channel complaint.

# 

# \---

# 

# \## 6. What is misleading about my headline number?

# 

# The headline \*\*86.96% escalation recall\*\* is useful, but it can be misleading if interpreted as evidence that the system is ready for autonomous support.

# 

# First, escalation recall is achieved partly through conservative routing. The system escalates \*\*132 of 180\*\* evaluated cases, leaving only 48 for auto-handling. Therefore, high recall comes with substantial escalation volume.

# 

# Second, the golden set is only 180 cases and is intentionally diversity-oriented. Its distribution may differ from the distribution of real incoming AmazonHelp traffic.

# 

# Third, escalation recall says nothing directly about whether the generated replies are correct or well grounded.

# 

# Finally, the system still produced \*\*12 false auto-handles\*\*, meaning some cases that humans considered escalation-worthy were routed to auto-handling.

# 

# \*\*The more honest takeaway is:\*\* the system is promising as a \*\*conservative support triage assistant\*\*, but the current evidence is insufficient to justify broad autonomous customer handling.

# 

# \---

# 

# \## 7. What I would do with one more week

# 

# 1\. \*\*Separate classification confidence from resolution confidence.\*\*

# &#x20;  A case can have a clear intent but still require account/order investigation. This is likely the highest-impact safety improvement.

# 

# 2\. \*\*Add multilingual semantic retrieval.\*\*

# &#x20;  Evaluate whether multilingual embeddings improve retrieval and intent performance on non-English/noisy messages.

# 

# 3\. \*\*Create an independent judge-human validation set.\*\*

# &#x20;  Have humans score a fixed subset of generated replies and measure agreement with the LLM judge before using judge scores as a quality metric.

# 

# 4\. \*\*Evaluate routing by intent and risk tier.\*\*

# &#x20;  Instead of only reporting aggregate escalation recall, measure false auto-handles separately for payment, account, delivery, technical, and other high-risk categories.

# 

# The priority should be reducing \*\*false auto-handles\*\* rather than maximizing raw intent accuracy.

# 

# \---

# 

# \## 8. Conclusion

# 

# The proposed AmazonHelp support agent substantially outperforms the trivial and lexical baselines on intent classification while achieving high escalation recall. Its main limitation is that historical similarity and intent confidence do not guarantee that a case can actually be resolved without account-specific investigation. The system is therefore best positioned as a conservative triage and drafting assistant today, with safer autonomous handling as the next engineering target.




