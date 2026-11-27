# Third-party inference provenance and licensing

Reviewed 2026-11-27. This is informational, not legal advice.

## What is actually published

The GitHub repository and the PyPI distribution contain **source code, configs,
tests, and documentation only**. Teacher-label JSONL files, SQLite caches,
trained student artifacts, and all `runs/` output are git-ignored and are not in
the sdist or wheel. No dataset or model derived from Groq or Gemini outputs has
been published as part of this release.

That materially limits the licensing surface. The published package does not
reproduce provider outputs; it contains code that can call a provider if the user
supplies their own key and config.

## Groq Cloud — `openai/gpt-oss-20b`

Governing documents reviewed:

- Groq Services Agreement, last modified 2026-06-22:
  https://console.groq.com/docs/legal/services-agreement/
- Groq Acceptable Use & Responsible AI Policy, effective 2025-10-15:
  https://console.groq.com/docs/legal/ai-policy
- Groq data handling / zero-data-retention notes: https://console.groq.com/docs/your-data
- Model page: https://console.groq.com/docs/model/openai/gpt-oss-20b
- OpenAI gpt-oss weights license: https://github.com/openai/gpt-oss/blob/main/LICENSE
- gpt-oss usage policy: https://github.com/openai/gpt-oss/blob/main/USAGE_POLICY

Relevant points:

- Groq's agreement treats Inputs and Outputs as Customer Data. Section 8.1 says
  the customer retains IP rights in Customer Data, including Outputs; section
  4.2 makes the customer responsible for use of Outputs and restricts **Groq**
  from training on them without permission.
- Section 4.1 makes applicable AI Model Terms control for use of a model. The
  gpt-oss weights are Apache-2.0 with a short usage policy; no output-specific
  training/publication ban was found there.
- The AUP bars illegal, infringing, harmful, deceptive, and abusive uses; the
  label-generation runs here were narrow classification tasks and did not alter
  provenance metadata or claim provider endorsement.
- Section 6.3 restricts developing competing offerings and altering transparency
  or provenance information. Bantam Cascade is not a Groq Cloud service.

Conclusion: on the public terms reviewed, no clause prohibits using these
outputs as teacher labels for this research. This is **not** a legal clearance;
private/enterprise terms may differ and terms can change. Provenance is recorded
here and no provider weights, logos, or trademarks are shipped.

## Google Gemini API — `gemini-decision-1` (Decisions API)

Correct endpoint: `POST https://generativelanguage.googleapis.com/v1beta/decisions`.
The model id `gemini-decision-1` is the documented Decisions API model code;
it is not a `gemini-2.x` text-generation id. At review time the published docs
and blog listed it while the generated v1beta discovery document lagged, so we
rely on the published documentation.

Governing documents reviewed:

- Decisions API guide: https://aistudio.google.com/docs/decisions-api
- Model page: https://aistudio.google.com/docs/models/gemini-decision-1
- REST reference: https://aistudio.google.com/api-reference/decisions/create-decision
- Announcement: https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/
- Gemini API Additional Terms, effective 2026-03-23:
  https://aistudio.google.com/docs/terms
- Google APIs Terms of Service: https://developers.google.com/terms
- Generative AI Prohibited Use Policy:
  https://policies.google.com/terms/generative-ai/use-policy
- Privacy Policy: https://policies.google.com/privacy

Relevant points:

- The Gemini API Additional Terms say Google will not claim ownership of
  generated content and that the user is responsible for its use and for sharing
  it. No clause specific to the Decisions API prohibits classification labels.
- The competing-model and reverse-engineering restrictions target the Services
  and their weights/components, not ordinary labels.
- For **unpaid** services, Google may use submitted content and generated
  responses to improve its products and human reviewers may read them. The local
  exploration did not record paid-quota assurance, so it must be treated as
  unpaid-quota data for provenance purposes.
- Grounding with Google Search/Maps has stricter training restrictions, but no
  grounding features were used.
- Google APIs Terms §5(e) contains broader language about permanent copies of
  content returned from APIs; the more specific Gemini Additional Terms are the
  better reading for ordinary generated content, but this supports a
  conservative posture.

Conclusion: no public clause clearly prohibits using `gemini-decision-1`
probabilities as teacher labels, but there is no express publication license
either. Because paid-quota status was not recorded, **do not publish Gemini-derived
labels or models from these exploratory runs**. If publication is desired,
re-generate under paid quota, record that fact, avoid grounding features, and
obtain legal sign-off.

## Secrets and history scan

`python scripts/scan_secrets.py --history` scans tracked files, all commits, and
reflogs for Groq `gsk_`, Gemini `AIza`, and generic assigned-key patterns. It
prints only pattern names, commits, and paths, never matched values. On
2026-11-27 it reported **0** findings across all tracked files, commits, and
reflogs scanned. Secrets live only in the local git-ignored `.env`.

## Position for this release

- No Groq- or Gemini-derived labels, caches, or student models are shipped.
- Teacher outputs remain local, git-ignored artifacts used only to measure the
  cascade and produce the reported numbers.
- All accuracy and certification claims are evaluated against the original
  datasets' gold human labels, not against teacher labels.
- No provider weights, logos, trademarks, grounded third-party content, or
  provenance metadata are redistributed.
- If someone later publishes derived labels or models, they must re-check the
  current provider terms, record quota status, preserve provenance, and avoid
  exclusivity or endorsement claims.
