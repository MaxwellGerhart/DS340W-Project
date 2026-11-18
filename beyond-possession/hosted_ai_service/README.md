# Shared Gemini service

This is the source for the shared Google Gemini service used by the public project download. Its published origin is https://beyond-possession-ds340w-ai.arnavbedi123.chatgpt.site . The project code and launchers remain in MaxwellGerhart/DS340W-Project.

The Worker runs on hosted infrastructure. Configure GEMINI_API_KEY, GEMINI_MODEL and SESSION_SECRET as server-side secrets. No secret values belong in this directory or in public Git history. SESSION_SECRET is a random signing key for short-lived, anonymous demo sessions.

The local Python backend obtains a scoped session from POST /api/session, then sends at most 24 metric records to POST /api/select. The service constructs the Gemini request itself, accepts only supported soccer analyst roles and metric fields, validates structured response fields, and returns observations to the existing numerical critic on the computer. It does not provide an unrestricted chat or URL proxy. GET /api/health checks service configuration without generating AI output.

POST /api/write supports bounded free writing with numeric bindings and at most eight correction-feedback records. POST /api/review accepts at most eight claims and 24 cited records, returning a complete set of tactical pass/fail verdicts. These endpoints use the same scoped sessions and rate limits, fixed server prompts and a 24 KB maximum request. Model-produced numbers are still checked by the Python client. Set optional server-side GEMINI_CRITIC_MODEL to use another accessible Gemini model for review; otherwise review is a separate call to the same base model. Writer/reviewer prompts and schemas match beyond_possession/narrative.py. See ../docs/APPROACH_CHANGES.md for the experimental modes and limits.

Sessions expire after 15 minutes. Per-IP session and generation limits and a one-hour response cache are held in each running Worker instance; they are best-effort demo controls, not a durable global spending limit. Google project quotas and billing still apply. The server never switches providers or retries quota errors automatically. Use direct Gemini mode with your own key for uncached research measurements.

Build with bash scripts/build.sh. Check with node scripts/test-worker.mjs and node scripts/validate-artifact.mjs. The deployed build is a Cloudflare Worker ESM at dist/server/index.js, with the hosting manifest copied into dist/.openai/hosting.json. Production secrets are configured separately from source.
