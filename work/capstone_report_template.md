# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Mohamed Ramadan
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/MohamedRamadan164/FlyRank_ML_internship
- **Date:** 2026-08-31

> Copied from `capstone_report_template.md` into `work/capstone_report.md` and filled in from
> the work already done in `w01`–`w05` and `capstone.ipynb`.

## 0. Abstract

Content teams cannot review every page every month, so this work asks which pages are worth
reviewing first. A transparent rule-based baseline (stale, still visible, underperforming its
position's usual click-through rate) is compared against a logistic-regression model, both
evaluated at precision@K on a client-grouped train/test split so results reflect clients the
model has never trained on. On a validated 30,000-row prototype sample, the baseline reaches
precision@50 of 0.40 against a 0.52 base rate, while the model reaches 0.62. The output is a
ranked, reason-coded action queue (`refresh` / `protect` / `monitor`) meant as decision support
for a human editor, not an automated verdict.

## 1. Problem framing

**Decision:** which content items in a client's catalog should an editor spend limited review
time on this month. **Unit of analysis:** one content item (`content_id` / `content_hash_id`),
summarized over a defined window — not a page-day, not a client. **Output:** a ranked score per
content item with one reason code and one action label (`refresh` / `protect` / `monitor`).
**Action a human takes:** opens the top of the queue and decides whether to rewrite, update, or
leave the page alone — nothing publishes automatically. **Cost of a wrong call:** asymmetric. A
false `refresh` flag wastes an editor's time on a page that was fine (cheap, reversible). A missed
decline (false `monitor`) is worse — a real problem goes unnoticed because nothing prompted a
check. **Why ML helps:** no single signal (staleness, CTR, position, volume) correlates with
decline on its own — all individual correlations with the trend proxy are under 0.05 in
magnitude (`w02`) — but combined signals separate decliners with ROC-AUC 0.70–0.75. That gap
between "no single rule works" and "combined signals do" is exactly the case for a learned model
over a hand-written threshold.

## 2. Data safety

**Starter-sample prototype:** `data/raw/content_refresh_anonymized.csv`, 30,000 rows, 32
pseudonymized clients, trailing-90-day metrics. **Full-scale (capstone) data:**
`FlyRank/internship-warehouse` v20260703 on Hugging Face — `dim_content` (519,606 rows) joined to
`fact_content_daily_performance` (78.8M rows, month-partitioned), analyzed on a mid-panel month
(`month=2026-03`), never the sealed final month.

**Deliberately excluded, and why:**
- `trend_direction` / `trend_pct` (starter CSV) — these define the label; `impressions_last_30d`
  / `impressions_prev_30d` correlate ~1.00 with `trend_pct` and are excluded for the same reason
  (checked directly in `w05`, not assumed).
- `clicks_second_half` / `impressions_second_half` (warehouse) — define the capstone's proxy
  label; used only to build the trap demonstration in `w03`, never as a feature.
- `fact_content_query_90d` — its 90-day window overlaps the analysis month's boundary; excluded
  until that alignment is done properly.
- Any GA4-derived column where `ga4_data_available` is not `TRUE` — zero-filled placeholders,
  not real zero engagement.
- `content_id` / `client_id` / `content_hash_id` / `client_hash_id` — pseudonyms, used only for
  joining and for the grouped train/test split, never as features.

**Confirmed:** no client name, domain, raw query, or credential appears anywhere in `work/` —
every notebook and this report use only pseudonymous IDs and aggregate statistics.

## 3. Baseline

Plain-English rule (from `w04`): *"A page is worth reviewing if it hasn't been touched in 90+
days, it's still getting real impressions, and its click-through rate is underperforming what
its position tier normally earns."* Coded as
`score = stale_flag × visible_flag × ctr_gap × impressions_90d`, one reason code
(`stale_ctr_underperform`), one action label (`review_for_refresh`). Two signals were checked
before writing it: staleness alone vs. decline rate was **MIXED** (51% → 61% → 47% across
freshness tiers, non-monotonic), CTR vs. position was **CONFIRMED** (weighted CTR falls
steadily from ~0.49% at top-3 to ~0.04% deep). Baseline precision@50 on the client-grouped test
split: **0.40**, against a base rate of 0.52 — a fair, honest, beatable starting point, evaluated
on the identical split and metric the model uses.

## 4. Model / analysis

**Method:** Logistic Regression (primary) compared to a shallow Random Forest (`max_depth=6`) —
chosen per the "yes/no with an observed-ish label" + "which first?" ranking shape: readable
first, add complexity only if it earns its keep (per `w05`, it didn't — logistic regression
matched or beat the forest at every K).

**Feature list (13 features):** `word_count`, `content_age_days`, `days_since_last_update`,
`avg_position`, `ctr`, `engagement_rate`, `scroll_rate`, `impressions_90d`, `clicks_90d`,
`search_volume`, `cpc`, `content_type`, `main_intent`. **Left out on purpose:** every
label-derived or window-overlapping column listed in Section 2.

**Target/proxy, in one sentence:** whether a content item's clicks fell from the first half to
the second half of the analysis window (`is_declining_this_month` / `trend_direction == "down"`
in the starter sample) — a same-window proxy, not an independently observed future outcome.

## 5. Evaluation

**Split:** grouped by client (`GroupShuffleSplit`, 75/25, `client_id`/`client_hash_id`), not a
plain row split — content items from a given client land entirely in train or entirely in test,
so precision@K reflects generalization to clients never seen during training, not memorized
per-client style. Verified programmatically that train/test client sets never intersect.

**Metrics vs. base rate (client-grouped test split, starter-sample prototype, base rate 0.517):**

| K | Base rate | Baseline rule | Logistic Regression | Random Forest |
|---|---|---|---|---|
| 20 | 0.517 | 0.40 | **0.65** | 0.60 |
| 50 | 0.517 | 0.40 | **0.62** | 0.56 |
| 100 | 0.517 | 0.46 | **0.60** | 0.59 |

**Errors:** the model's highest-confidence false positives (declared "declining," actually
`up`/`stable`) all sit at `days_since_last_update` in the 92–104 day range — the same ambiguous
staleness band flagged as MIXED in Section 3. A recently-published page and a genuinely
neglected page can carry an identical "104 days since last touch" value; the model hasn't
resolved that ambiguity, it's weighing it better than a hard threshold does, not eliminating it.

## 6. Interpretation

Top permutation-importance feature (ROC-AUC scoring): `impressions_90d`, followed by
`content_age_days` and `avg_position`. This is plausible, not a leakage flag — it's a 90-day
volume total, not a future-window value. **Surprise / negative result worth keeping:** staleness
alone is a weak, non-monotonic signal (Section 3's MIXED verdict) — an intuitive "older pages
decline more" rule would be wrong at the tail (181+ days actually shows the *lowest* decline
rate in the sample, n=174, small but honestly reported rather than dropped). The honest reading
is that staleness only becomes useful combined with visibility and CTR-gap, not on its own.

## 7. Recommendation

Three reason-coded actions an editor can act on tomorrow:

- **`refresh`** (`stale_ctr_underperform`) — stale, visible, underperforming CTR for its
  position tier. Highest-priority review.
- **`protect`** (`growing_leave_alone`) — already well-positioned and trending up. Recommendation
  is to leave it alone.
- **`monitor`** (`ambiguous_watch`) — no clear signal either way. Light periodic check, not
  immediate action.

On the prototype sample this produced 4,129 `refresh` / 1,401 `protect` / 24,470 `monitor` items
out of 30,000 — a workable shortlist, not the whole catalog. **Confidence and limits, stated
explicitly:** this is decision-support built on a same-window proxy label and a single mid-panel
month; it does not claim any recommendation, once acted on, will cause a traffic or ranking
change, and it makes no claim about Google's algorithm itself — see `capstone_paper.html`
Section V (Limitations) for the full list.

## 8. Reproducibility

**Re-run from a fresh clone:**
```bash
git clone https://github.com/MohamedRamadan164/FlyRank_ML_internship.git
cd FlyRank_ML_internship
pip install -r requirements.txt
python scripts/run_all.py                 # starter-sample pipeline sanity check
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w01_research_question.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w02_ml_task_framing.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w03_data_contract.ipynb   # needs HF_TOKEN
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w04_baseline_score.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w05_model.ipynb
jupyter nbconvert --to notebook --execute --allow-errors --inplace work/notebooks/capstone.ipynb  # 🌐 cells need HF_TOKEN
```
**Random seed:** 42, fixed everywhere a split or a model is fit. **Environment:** `pandas`,
`numpy`, `scikit-learn`, `duckdb` (warehouse cells only), `matplotlib` — see `requirements.txt`
for exact pins. **Sealed/holdout evaluation status:** the client-grouped test split is built and
scored inside `w05_model.ipynb` and `capstone.ipynb` themselves (the split cell + the metrics
table are in the same notebook, same run) — not a separate untracked script, so "evaluated once,
on a held-out client set" is checkable directly from the committed notebooks, not taken on faith.
The full-warehouse run (`capstone.ipynb` Section 4b) is written but requires a personal
`HF_TOKEN` to execute — **not yet run**, its cells still carry `[[FILL IN AFTER YOU RUN]]`
placeholders pending that run.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai)

---

> **Claims checklist before submitting:** every number above is observed / measured / directional
> / decision-support language. No causal claims — no experiment exists in this data. No claim of
> predicting Google's algorithm — the label is a proxy computed from FlyRank's own panel. No
> client-identifying details anywhere. Base rate (0.517) is reported next to every precision@K so
> a reader can judge the numbers against chance, not in isolation. **Not yet true:** "numbers in
> this report match a fresh re-run" — that's only fully true once Section 4b's warehouse cells are
> run with a real `HF_TOKEN` and the `[[FILL IN AFTER YOU RUN]]` placeholders are replaced.
