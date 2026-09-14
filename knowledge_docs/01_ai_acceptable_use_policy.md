# AI Acceptable Use Policy

## Purpose and Scope
This policy governs the use of any artificial intelligence (AI), machine learning (ML), or automated conversational tool by any employee, contractor, or intern, on any company-owned or personally-owned device used for company work. It applies regardless of how the AI is accessed: a website, a browser extension, a desktop or mobile app, a code editor plugin, an API call, or a chat widget embedded in another product. If a tool sends any input to a model hosted outside the company's own infrastructure, this policy applies.

## Approved AI Tools
The following AI tools are approved for business use, subject to using the company enterprise-licensed account (personal or free accounts are NOT covered by this approval):
- Microsoft Copilot (Microsoft 365 Copilot, enterprise tenant only)
- GitHub Copilot (organization-licensed seats, engineering team only)

No other AI tool is approved for company data at this time. Approval status changes periodically; the security team maintains the current list.

## Unapproved Generative AI Chat Tools (Shadow AI)
The following consumer/generative AI chat and assistant tools are NOT approved for company or customer data, regardless of which account (personal or free tier) is used:
ChatGPT (chatgpt.com, chat.openai.com), Claude (claude.ai, api.anthropic.com), Google Gemini / Bard (gemini.google.com, bard.google.com), Perplexity (perplexity.ai), Cohere (cohere.ai, cohere.com), Mistral (mistral.ai), Poe (poe.com), Character.AI (character.ai), DeepSeek (deepseek.com), Grok / x.ai (x.ai, grok.com), Replicate (replicate.com), Together AI (together.ai), Groq (groq.com), Fireworks AI (fireworks.ai), Hugging Face (huggingface.co), Jasper AI (jasper.ai), Writesonic (writesonic.com), Copy.ai (copy.ai), ElevenLabs (elevenlabs.io), Runway ML (runwayml.com), Midjourney (midjourney.com), Stability AI (stability.ai), Cursor (cursor.com).

Use of any of these tools with company data — including pasting text, uploading a file, or sending a prompt containing business information — is a policy violation.

## Browser Extension AI Assistants
Browser extensions that add an AI sidebar or "ask AI" button to every webpage are treated the same as the underlying model they proxy to, and are NOT approved:
Monica AI Extension (monica.im), Merlin AI Extension (getmerlin.in), Sider AI Extension (sider.ai), MaxAI Extension (maxai.me), Wiseone Extension (wiseone.io), AskYourPDF Extension (askyourpdf.com).

These tools are considered higher risk than direct website use because they can read the content of every page the employee visits, including internal web applications, and may summarize or transmit that content to an external AI backend without an explicit "send" action.

## Developer Environment AI Assistants
AI coding assistants embedded in an IDE or terminal are only approved as listed under Approved AI Tools above (GitHub Copilot, engineering only). The following are NOT approved for use on company source code, credentials, or infrastructure configuration: Codeium (codeium.com), Windsurf (windsurf.com), Tabnine (tabnine.com), Sourcegraph Cody (sourcegraph.com), Continue.dev (continue.dev), Supermaven (supermaven.com), JetBrains AI Assistant (ai.jetbrains.com), Replit AI (replit.com).

Source code is Confidential data (see Data Classification & DLP Standard). Sending source code to any unapproved AI coding assistant is a Shadow AI incident, not just a tooling preference.

## Chatbot and Live-Chat Platforms
Third-party chat-widget and conversational-AI platforms — Intercom, Drift, Zendesk Chat / Zopim, Tidio, Crisp, LiveChat, Freshchat, ManyChat, Chatfuel, Landbot, Kore.ai, Yellow.ai, Haptik, Ada, Verloop, MobileMonkey, Botpress, Voiceflow, Dialogflow, IBM Watson Assistant, Rasa, Chatbase — are common in customer-support tooling and may be legitimately embedded in the company's own customer-facing products. Their presence in logs is not automatically a violation.

They become a Shadow AI finding when: (a) an employee uses one of these tools to process internal or confidential company data rather than customer-facing support content, (b) the platform is not on the approved vendor list maintained by IT/Legal, or (c) the traffic pattern shows bulk data (see thresholds in the Data Classification & DLP Standard) flowing to the platform outside of normal support-ticket volume.

## Cloud and API-Based AI Services
API-level AI usage is governed the same as chat-tool usage. This includes: AWS Bedrock (bedrock-runtime, invoking any foundation model including Anthropic, Amazon Titan, or Meta models), AWS SageMaker endpoints, AWS Comprehend, Rekognition, Textract, Polly, Lex, Transcribe, Personalize, Kendra, Forecast, Amazon Q Developer (CodeWhisperer), and Azure OpenAI Service (openai.azure.com). Business use of any AWS or Azure AI/ML service requires prior approval and must go through a company-owned account with an active data processing agreement — never a personal or unmanaged cloud account.

## Catch-All for Unlisted AI Tools
This policy cannot name every AI tool that will ever exist. Any application, browser extension, IDE plugin, API, or chat widget that is functionally a generative AI, large language model, or AI-based chatbot or assistant — and is not explicitly named on the Approved AI Tools list above — is treated as UNAPPROVED by default, even if it is not yet named in this document. Security monitoring that surfaces an unrecognized external domain exhibiting AI-like traffic patterns will be evaluated against this default-deny rule until formally reviewed and added to either the approved or unapproved list.

## Prohibited Actions for All AI Tools
Never enter customer PII, credentials, API keys, or financial data into any AI tool's prompt or file upload, even an approved one, unless the specific approved deployment has been certified for that data type. Never paste or upload proprietary source code to an unapproved AI coding assistant. Never use a personal or free-tier account of an otherwise-approved vendor, such as a personal ChatGPT Plus account, for company work — only the enterprise-licensed deployment is covered. Never disable, bypass, or attempt to evade network monitoring, proxy filtering, or DLP controls in order to reach an AI service.
