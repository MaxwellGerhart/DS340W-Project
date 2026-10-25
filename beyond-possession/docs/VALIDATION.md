# Validation performed

- Python 3.14.3: 20 automated unittest checks passed.
- Synthetic CSV and JSON regenerated successfully with seed 42.
- CLI analysis from files produced the included Markdown and JSON example reports.
- All six experiment systems ran twice: 12 successful runs and zero failures.
- Blinded reports and blank forms for three reviewers were generated.
- SQLite persistence and numerical/reference rejection were verified.
- HTTP request handlers were exercised directly, including session-token rejection, static routing, demo analysis and malformed uploads.
- JavaScript syntax and seven dashboard render paths were checked using JavaScriptCore with a minimal DOM test harness. Missing-tracking rendering and HTML escaping also passed.

The execution workspace prevented binding a local server port. A live browser layout and interaction check was therefore not performed. Start the dashboard locally and check layout, filters, file selection and downloads before presenting it.

Ollama request formatting, structured output parsing and token logging were tested with mocked HTTP responses. A running model was not available, so actual local inference and model quality remain unverified. GitHub CI is configured for Python 3.11 through 3.14; those other versions were not run locally.

The included examples are synthetic workflow demonstrations. No real-match research result or expert score is claimed.

Match catalog update: 25 total automated checks now pass, including seven-file merges, original-field preservation, provenance, duplicate handling, conflicting-ID rejection, invalid JSON and input/output protection. These checks use fixtures; the seven actual attachments were inaccessible and are not included.

VS Code update: added portable workspace, debug configurations, tasks, unittest discovery, extension recommendations and a run.py launcher that opens the browser only after the HTTP server binds successfully. Configuration JSON and the launcher help command were checked. VS Code UI execution itself was not available in this workspace.

Ollama diagnostics update: 30 automated checks pass. New cases cover unreachable service, missing model, successful model lookup, preservation of memory-error response text and malformed model JSON. The app now preflights the configured model, runs Ollama specialists sequentially, and provides a Check Ollama button plus a doctor command. Actual model inference remains unverified because Ollama is not installed in this workspace.

Direct local update: found the seven extracted match JSON files under Downloads/combined_datasets. Inspected their actual records and merged 1,941 matches with zero duplicate IDs into data/real/combined_matches.json. The dashboard loads this catalogue automatically, supports searching all matches, and can merge uploaded match JSONs. Event uploads now accept either normalized CSV or normalized JSON. All 34 tests passed; JavaScript execution checks covered section highlighting, chart rendering, real catalogue rendering and search. Opened the edited project in VS Code at Documents/ChatGPT/DS 340 W/beyond-possession. A live browser test of this update was not performed; VS Code opened the new folder in Restricted Mode.

Automatic AI setup update: 44 automated checks passed. Fixture tests verify platform asset selection, SHA-256 verification, reuse of cached runtime/model, streamed model download completion, failure cleanup, reuse of an existing service and preservation of an explicit endpoint. Frontend JavaScript execution checks and offline analysis also passed. Attempting setup_ai.py reached the official runtime-download step, but DNS/network restrictions prevented the actual download. No real model inference or native Windows/Linux installation was verified. The local AI binaries and weights are not included in the source package.

Groq API update: 54 automated checks passed. Coverage includes dotenv loading without code execution, credential headers, strict JSON schema requests, token logs without secrets, rate-limit behavior, key redaction, incomplete outputs, pipeline evidence checks, non-inference connection checks, TLS verification, and API startup skipping Ollama downloads. The frontend render/navigation checks passed. A blank .env file is open in VS Code for the user's own key; it is excluded from Git and the saved source package. No live hosted inference was run because no user API key is configured.

Quota protection update: 59 tests pass, including bounded prompts with team/interval coverage, no second inference call for a cached dashboard request, uncached research repetitions, immediate stop after first specialist failure, and redacted provider errors with retry timing. Frontend rendering checks passed. No live Groq inference was run to diagnose the user account; the exact rejected request remains unknown until the provider message is supplied.

Local-default update: switched the local .env and launcher defaults to Ollama, preserving any existing API key without printing or packaging it. Extended verified TLS certificate-bundle handling to all runtime/model-download requests. All 59 tests and frontend execution checks pass. An actual runtime download was attempted, but the workspace DNS restriction prevented completion. Local model installation and inference remain unverified.
