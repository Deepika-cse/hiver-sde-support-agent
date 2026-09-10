# \# Golden Set Labeling Guide — AmazonHelp

# 

# \## Goal

# 

# Create a human reference set of 150–250 AmazonHelp customer-support cases.

# 

# For this project, the Golden Set contains 180 cases sampled from the reconstructed AmazonHelp support population in the Customer Support on Twitter (TWCS) dataset.

# 

# Labels must be based on:

# 1\. The latest customer message.

# 2\. Relevant previous turns in the conversation.

# 3\. What can reasonably be inferred from the customer's request.

# 4\. The historical AmazonHelp response when evaluating whether a grounded response is possible.

# 

# Do not infer facts that are not supported by the conversation.

# 

# \---

# 

# \## Intent Labeling

# 

# Choose exactly one `gold\_intent` from `configs/taxonomy.yaml`.

# 

# Use the customer's PRIMARY need.

# 

# If two intents genuinely coexist, use the most important customer need as `gold\_intent` and record the other as `gold\_secondary\_intent`.

# 

# Do not assign an intent merely because a keyword appears.

# 

# \### 1. order\_issue

# 

# Use when the main problem concerns an order itself.

# 

# Examples:

# \- Problem with an order after placing it.

# \- Incorrect order information.

# \- Questions about an order that are not specifically about delivery.

# \- General order-status questions when there is no clear delivery problem.

# 

# Do NOT use this when the primary issue is clearly a delayed delivery, delivered-but-not-received package, refund, or cancellation.

# 

# \---

# 

# \### 2. delivery\_issue

# 

# Use for:

# \- Late deliveries.

# \- Delivery delays.

# \- Expected delivery dates.

# \- Shipping delays.

# \- Tracking or shipment progress.

# \- Questions about when a package should arrive.

# 

# Examples:

# \- "My package is two days late."

# \- "Where is my package?"

# \- "It was supposed to arrive today."

# 

# If tracking says DELIVERED but the customer says they did not receive it, use `package\_not\_received` instead.

# 

# \---

# 

# \### 3. package\_not\_received

# 

# Use when:

# \- Tracking says delivered but the customer says the package was not received.

# \- The package was marked delivered to the customer but is missing.

# \- The customer explicitly disputes a delivery confirmation.

# 

# Examples:

# \- "It says delivered but I don't have it."

# \- "Amazon says it was delivered Saturday but I never got it."

# 

# This intent takes priority over general delivery issues when a delivered-but-missing situation is explicit.

# 

# \---

# 

# \### 4. return\_cancellation

# 

# Use for:

# \- Cancelling an order.

# \- Returning a product.

# \- Questions about return procedures.

# \- Problems with an existing return.

# 

# Examples:

# \- "I want to cancel my order."

# \- "How do I return this?"

# \- "I already returned the item but need an update."

# 

# If the main issue is receiving a refund after a return, use `refund\_payment`.

# 

# \---

# 

# \### 5. refund\_payment

# 

# Use for:

# \- Missing refunds.

# \- Refund status.

# \- Incorrect charges.

# \- Duplicate charges.

# \- Payment problems.

# \- Money-related issues.

# 

# Examples:

# \- "Where is my refund?"

# \- "I was charged twice."

# \- "When will I get my money back?"

# 

# If the customer only wants to return/cancel something and does not primarily ask about money, use `return\_cancellation`.

# 

# \---

# 

# \### 6. prime\_subscription

# 

# Use for:

# \- Amazon Prime membership.

# \- Prime trials.

# \- Prime renewal.

# \- Prime subscription charges.

# \- Prime benefits.

# 

# Examples:

# \- "Why was I charged for Prime?"

# \- "I want to cancel Prime."

# \- "Why did my Prime trial end?"

# 

# If the customer reports unauthorized activity or account compromise related to Prime, use `account\_issue` or another applicable intent and escalate.

# 

# \---

# 

# \### 7. account\_issue

# 

# Use for:

# \- Account access.

# \- Login problems.

# \- Password problems.

# \- Account settings.

# \- Account closure.

# \- Account verification.

# \- Account-specific access problems.

# 

# Examples:

# \- "I can't log into my account."

# \- "My password isn't working."

# \- "I want to close my Amazon account."

# 

# If the customer reports unauthorized transactions, account takeover, or suspicious activity, classify according to the primary account/security problem and mark it for escalation.

# 

# \---

# 

# \### 8. product\_technical\_issue

# 

# Use for:

# \- Product functionality problems.

# \- Amazon device problems.

# \- Fire TV, Echo, Kindle, or similar technical issues.

# \- Product-specific technical problems.

# 

# Examples:

# \- "My Fire TV Stick isn't working."

# \- "The sound and picture aren't synchronized."

# \- "My Echo isn't working."

# 

# \---

# 

# \### 9. complaint\_followup

# 

# Use when the PRIMARY request is about an unresolved previous support interaction.

# 

# Examples:

# \- "I contacted customer service three times and nobody fixed it."

# \- "You said this was escalated but I still haven't heard back."

# \- "I was promised an update yesterday."

# 

# Do not use this simply because the customer is angry.

# 

# If the customer is angry but clearly asks about a delivery, refund, order, etc., classify according to that actual need.

# 

# \---

# 

# \### 10. other

# 

# Use when the request genuinely does not fit the defined intents.

# 

# Examples:

# \- Vague requests with insufficient information.

# \- General questions that cannot reasonably be mapped to another intent.

# \- Non-support messages.

# 

# Use this sparingly.

# 

# \---

# 

# \# Secondary Intent

# 

# `gold\_secondary\_intent` is optional.

# 

# Leave it blank unless another defined intent is genuinely present.

# 

# Example:

# 

# Primary:

# `return\_cancellation`

# 

# Secondary:

# `refund\_payment`

# 

# Only use a secondary intent when both needs are materially present.

# 

# Do not use secondary intent just because another topic is mentioned incidentally.

# 

# \---

# 

# \# Escalation Labeling

# 

# Set:

# 

# `gold\_should\_escalate=true`

# 

# when a trustworthy automated response should NOT be sent without human involvement.

# 

# Escalate cases involving:

# 

# 1\. Security or account compromise.

# 2\. Unauthorized or suspicious transactions.

# 3\. Fraud.

# 4\. Legal concerns.

# 5\. Safety concerns.

# 6\. Highly account-specific actions that cannot be safely completed from the available evidence.

# 7\. Conflicting or unreliable historical evidence.

# 8\. Cases where the correct resolution cannot be grounded in historical AmazonHelp responses.

# 9\. Situations where automation would require inventing information or promising an unsupported outcome.

# 

# Examples:

# 

# \- "Someone is trying to buy things using my account."

# \- "My account has been hacked."

# \- "I was charged for something I didn't authorize."

# 

# \---

# 

# \## Do NOT Escalate Merely Because:

# 

# \- The customer is angry.

# \- The customer uses profanity.

# \- The customer complains about poor service.

# \- The customer is impatient.

# \- The customer asks for a normal delivery update.

# \- The customer asks a normal refund question that can be grounded in evidence.

# 

# Emotion alone is NOT an escalation reason.

# 

# \---

# 

# \# Escalation Reason

# 

# If `gold\_should\_escalate=true`, write a short specific reason in:

# 

# `gold\_escalation\_reason`

# 

# Good:

# 

# > "Customer reports unauthorized activity on their account; human verification is required."

# 

# > "Historical evidence does not establish that this customer is eligible for the requested refund."

# 

# Bad:

# 

# > "This is complicated."

# 

# > "Customer is angry."

# 

# If escalation is false, leave the reason blank.

# 

# \---

# 

# \# Reply Notes

# 

# Write one short sentence in `gold\_reply\_notes`.

# 

# The note should describe what a good grounded response should contain or avoid.

# 

# Examples:

# 

# > "Should acknowledge the delivery delay and provide only a supported tracking next step."

# 

# > "Should not promise a refund because historical evidence does not establish eligibility."

# 

# > "Should ask for the missing order identifier before giving account-specific guidance."

# 

# > "Should direct the customer to human support because the issue involves unauthorized account activity."

# 

# > "Should explain the supported troubleshooting step without inventing product capabilities."

# 

# \---

# 

# \# Sampling Note

# 

# The Golden Set was sampled from the reconstructed AmazonHelp customer-support population in the Customer Support on Twitter (TWCS) dataset.

# 

# Target size: 180 cases.

# 

# The sampling script favors diversity rather than simply selecting the first 180 rows. The sampled cases should cover different conversation lengths, customer messages, and support situations.

# 

# Excluded cases should include cases that cannot be meaningfully labeled because the customer message is unavailable or unusable.

# 

# The final report should document:

# \- Source population.

# \- Target size.

# \- Sampling/diversity method.

# \- Exclusions.

# \- Labeling process.

# \- Label version.

# \- Whether labels were reviewed/adjudicated.

# 

# \---

# 

# \# Labeling Process

# 

# For each case:

# 

# 1\. Read `latest\_customer\_text`.

# 2\. Read `conversation\_context` when available.

# 3\. Identify the customer's primary need.

# 4\. Assign exactly one `gold\_intent`.

# 5\. Add `gold\_secondary\_intent` only when genuinely applicable.

# 6\. Decide whether a trustworthy automated response is safe.

# 7\. If escalation is required, write a specific `gold\_escalation\_reason`.

# 8\. Write a concise `gold\_reply\_notes`.

# 9\. Set `labeler` to the human labeler's identifier.

# 10\. Keep `label\_version` as `v1`.

# 

# Labels should be based on the evidence available in the case, not assumptions about what Amazon might normally do.

# 

# \---

# 

# \# Adjudication Principle

# 

# Ambiguous cases should be reviewed using the following priority:

# 

# 1\. Explicit customer request.

# 2\. Latest customer message.

# 3\. Relevant conversation context.

# 4\. Historical support evidence.

# 5\. Conservative escalation when the correct automated action cannot be established.

# 

# When evidence is insufficient to safely automate the resolution, prefer escalation rather than inventing a resolution.

# 

# \---

# 

# \# Quality Principle

# 

# The Golden Set is the human reference standard for this project.

# 

# The goal is NOT to make the agent appear artificially accurate.

# 

# Difficult, ambiguous, multilingual, adversarial, and unresolved cases should remain in the evaluation set when they naturally occur in the sampled population.

# 

# This makes the reported performance more honest and useful.

