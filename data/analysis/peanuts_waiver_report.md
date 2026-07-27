# Hello Peanuts' Waiver Edge: Final Verdict

## Verdict

**It is a real, repeatable, league-leading waiver edge â€” but roughly one-third the size the headline SPOE number implies. The edge is a volume/role-arbitrage engine (find undervalued everyday hitters, promoted rookie starters, and relievers seizing the save role; then hold them). It is NOT streak-timing skill, and it is partly squandered by excessive drops/churn.** Net: mostly skill on the *existence* of the edge (high confidence), mostly artifact/overstatement on its *magnitude* (medium-high confidence).

## Reconciling +78 SPOE (#1) vs -3.47 delta_roto (near-last, 8th/10)

The two numbers contradict only because they answer different questions and each is biased in an opposite direction:

- **SPOE (+78, #1 by 4.6x over SkeleThor's +17)** is a *backward-looking outcome* measure: realized production over the owned window minus a **frozen 2026-03-30 preseason projection**. It credits every realized outcome â€” breakouts, hot-streak variance, and the mechanically depressed baselines of undrafted players â€” to the manager, regardless of whether the pickup beat the alternative. It **overstates** decision value.
- **delta_roto (-3.47, 8th/10)** is a *forward-looking counterfactual*: each move scored against **current** ROS projections (the best available alternative at decision time). Its problem is that the team total blends adds and drops.

The decisive reconciliation: **his adds alone net +3.20 delta_roto (positive)** against current projections. The team total of -3.47 comes almost entirely from the **drop/churn side** â€” a ~-6.7 gap driven by dropping players who then produced (e.g., the -4.17 Ryan Weathers move) plus the cost of 34 transactions. So "near-last" is a **drop-management / over-trading story, not an evidence that his pickups were bad.** Both headlines are misleading: +78 is inflated ~3x; -3.47 is dragged negative by an unrelated churn problem.

## The Mechanism (what he actually does)

Decomposing all 23 waiver adds by *why the player was available* accounts for essentially 100% of the +25.08 SGP waiver over-performance across three repeatable buckets:

| Bucket | SGP | Examples |
|---|---|---|
| Everyday / breakout hitters ascending into full-time roles | +13.2 | Jordan Walker (303 PA, +3.74), Dillon Dingler (276 PA catcher, +3.65), Casey Schmitt (+2.48), Brandon Marsh (+2.25) |
| Young SP promoted into a rotation | +6.6 | Braxton Ashcraft (+2.87), Max Meyer (+1.87), Tolle (+1.53) |
| Relievers who seized the save role | +5.5 | Louis Varland (13 SV, +3.19), Bryan Baker (14 SV, +1.98), Soto (8 SV) |
| Speculative stashes / 1-week streamers / vets / injured | -0.25 | Vargas, Goldschmidt, Weathers, Stanton, Westburg (0 PA) â€” **net zero** |

The value is **banked through volume**, not rate: over_perf correlates with days_owned (r=0.68) and PA/IP (r=0.69). Star adds ran ~3.5 PA per owned-day (Walker 3.56, Dingler 3.54, Marsh 3.53) â€” full-time deployment, so bench inflation is not the lever.

## What is skill vs. variance vs. artifact

**Repeatable skill (the core of the edge):**
- **Role/playing-time arbitrage.** The cleanest signal: Varland and Baker were projected as ~1.6-SGP middle relievers preseason but banked 13 and 14 saves â€” the over-performance *is* the role change by definition, independent of rate luck. Same for rookie-SP promotions.
- **Breadth and consistency.** 21/23 adds (91%) produced positive SGP; 17/23 (74%) beat expectation. Herfindahl = 0.095 â†’ **effective_n â‰ˆ 10.5 independent contributors** (top add only 13.8%, top-3 ~39%). Split-half reliability holds: H1 mean over_perf +0.914, H2 +1.105, 91% positive rate in *both* halves. Variance concentrates and decays; this is distributed and stable â€” the signature of a process.
- **A hard downside filter.** n_cold_at_pickup = 0 on all 23 adds â€” he never buys a slumping player.

**Backward-looking variance (will not persist):**
- **The hot-hand timing channel does not work as a value source.** corr(pre-pickup prod, post-pickup prod) = -0.007; corr(n_hot_at_pickup, over_perf) = -0.05. 9 of 10 hitter adds regressed (mean prod_per_pa 0.414 â†’ 0.314, -24%); the hottest buys collapsed hardest (Eldridge .500â†’.194, Vargas .532â†’.295, Goldschmidt .469â†’.271). His best add, Dingler (+3.65), was **cold** at pickup (n_hot=0). Hotness is a selection *screen*, not the edge.
- **~5 SGP of unsustainable reliever rate stats.** Varland's 1.24 ERA/29 IP and Baker's 1.29 ERA/21 IP will regress even though the save *role* is real.

**Measurement artifact (the magnitude inflation):**
- Frozen preseason baselines are depressed for undrafted players, so **47% of waiver production reads as "over" almost by construction** (actual 53.07 vs expected 27.99).
- Production is measured over **owned windows including bench games** â€” an upper bound vs. Yahoo started-only standings.
- Windows begin at pickup, near local hitter peaks, folding hot-streak carryover into "value."

## Quantifying real vs. illusory

- **Real, calibrated edge:** waiver_over +25.08 + core_over +2.30 = **~+27.4 SGP total**. On this scale the league gap shrinks from **4.6x (SPOE) to ~1.5x** (waiver_over +25.08 vs #2 Jon's Underdogs +16.53). This is genuinely #1 and broadly sourced.
- **Illusory / overstated:** the +78 SPOE headline overstates real added value by **roughly 3x** (~+50 of the +78 is baseline depression + owned-window/bench inflation + faded streak carryover).
- **Squandered:** ~-6.7 delta_roto leaked back through drops/churn, which is what pushes the transaction analyzer to near-last despite the adds netting +3.2.
- Rough split: **~55-60% of the apparent value is real and repeatable** (role/PT arbitrage, volume, breadth); **~40-45% is artifact or already-faded variance**.

## Confidence and top caveats

**Confidence: medium-high** that a real repeatable edge exists; **medium** on the exact size and skill/luck split.

1. **One 52%-elapsed season, n=23 adds** (n=10-12 hitters with pre/post data). Correlations are directionally clear but statistically fragile â€” do not over-read magnitudes like the -0.775 reversion.
2. **Owned-window/bench inflation cannot be fully removed** without game-log start/sit data; every "actual" figure (including the 53.07 and +25.08) is an upper bound vs. true standings impact.
3. **The frozen-baseline confound is unresolvable here:** beating a stale preseason bar rewards generic under-projection of breakout youth (Walker, Dingler) as much as manager-specific skill. Continuation probabilities were unavailable, so persistence is inferred from raw pre/post production only.

## Actionable takeaway

**Copy the engine, ignore the theater, and fix the leak.**
- **Copy:** the repeatable core â€” target players gaining *role/playing-time* (bullpen closer shifts, rotation promotions, everyday-lineup ascensions), and **hold them to bank volume**. This is the teachable, scalable edge and it is independent of luck.
- **Ignore:** the "he times hot streaks" narrative â€” the timing signal has zero forward predictive power. Chasing hot rates off 34-56 PA windows is regression-chasing; the never-add-cold screen is fine as downside protection but is not itself an alpha source.
- **Fix / defend against:** his own churn. His adds are net-positive forward (+3.2 delta_roto) but he bleeds it back via drops. The lesson for him â€” and the vulnerability for opponents â€” is that the acquisition skill is real; the value destruction is on the drop side. Don't over-trade productive players away.

Bottom line: **the truth sits well above -3.47/near-last and well below +78/#1-by-4.6x** â€” a strong, broad, mostly-repeatable ~+27 SGP waiver edge built on role/playing-time arbitrage, materially overstated by the SPOE headline and partially squandered through churn.