# Data Classification and DLP Standard

## Data Classification Levels
Confidential data includes source code, credentials, API keys and secrets, financial records, unreleased product plans, customer PII, employee PII, and security findings or vulnerability data. Internal data includes internal process documentation, non-public internal communications, and aggregated non-PII business metrics. Public data is anything already publicly released by the company, such as marketing material and published documentation.

## Rules by Classification
Confidential data must NEVER be sent to any AI tool that is not on the Approved AI Tools list, and even on an approved tool, only through the certified enterprise deployment. Internal data should not be sent to unapproved AI tools; minor amounts in an approved tool are permitted for routine work. Public data has no AI-usage restriction.

## Bulk Transfer and Exfiltration Thresholds
Any single outbound transfer of more than 100 KB (102,400 bytes) to an unapproved AI or chatbot endpoint is treated as a potential bulk data exfiltration event and must be escalated regardless of what the data appears to be, since payload contents are frequently not visible in network logs. Any transfer exceeding 1 MB is CRITICAL severity by default. Cumulative transfers from the same user to the same AI service exceeding 500 KB within a 24-hour window are treated as CRITICAL even if no single transfer crossed the threshold alone.

## Special Handling for Browser Extensions and IDE Assistants
Because browser-extension and IDE-based AI assistants often transmit page or file content automatically rather than through a single explicit upload action, byte-count thresholds alone may under-count real exposure. Any confirmed use of an unapproved browser-extension or IDE AI assistant on a device with access to Confidential data is treated as at least MEDIUM severity even with a small logged byte count, escalating to HIGH if the browsing or editing history for that session included a Confidential-classified resource.
