# Identification-queue order: protocol, first run in British Columbia (draft for co-design, not sent)

Status: draft for the BC-team meeting. The set-up below is a proposal. Nothing here is final until Nathan, Laura and David have said yes.

## The question

Does the order in which the iNaturalist identification queue shows records change how much gets identified, and what? The method is general: split the records that need an ID into lists at random, change only the order each list shows, let identifiers work all lists in turn, and read the result back from iNaturalist. The first run is a British Columbia identification blitz with Blitz the Gap. Its results hold for that blitz's region and identifiers; other regions have to run it to know.

Orders answer one of two questions:

- **Speed:** does the order change how many IDs an hour of effort gives? Example: look-alike photos together.
- **Value:** does the order change which gaps the IDs fill? Examples: records from data-poor places first, which targets the gap in where species are known to occur (the Wallacean shortfall, [Hortal et al. 2015](https://doi.org/10.1146/annurev-ecolsys-112414-054400)); unfamiliar photos first, which targets the gap in what verified photos cover.

Newest first, close to what iNaturalist shows today, is the control for both.

## Proposed set-up for BC

- Four lists: newest first (the control), data-poor places first, look-alikes together, and unfamiliar photos first. An earlier draft of this protocol proposed two lists, newest first and data-poor places first, and left the other two to a later round. The four-list design replaces it.
- Four weeks in November 2026.
- All ten iNaturalist taxon groups.
- Participants join an iNaturalist project, so the analysis knows who took part.
- The page is hosted on the PollockLab GitHub Pages site and works in any web browser; a laptop works best, because Identify is built for a large screen.

## What identifiers see

One web page. An identifier picks a taxon group they know and presses "Next batch". Each press opens iNaturalist's Identify tool on a batch of about 120 British Columbia records that need an ID and have a photo. Identifiers work inside iNaturalist as usual: their IDs carry their name, count toward Research Grade, and flow to GBIF. Nothing about identifying changes. We only choose which records sit together in a batch.

Each browser draws its own random cycle of the lists once, and every press takes the next list in that cycle, so every identifier's batches split evenly over all lists and nobody's effort lands on one list only. The page shows only "Batch N". A few identifiers make most of the IDs, and if each identifier kept one list, the list that drew the busiest identifier would win whatever its order; the rotation borrows the within-person comparison of interleaved search evaluation ([Chapelle et al. 2012](https://doi.org/10.1145/2094072.2094078)).

The lists are unlabelled rather than blind. The page never names them, but an identifier who looks can often tell a list by its batches, most clearly with look-alike photos. This is why the test compares each identifier with themself and does not depend on anyone not knowing.

Identify shows records newest-first and lets people skip freely, so the batches control composition, not sequence. Records an identifier has already reviewed drop out of the batch automatically, and records that reach Research Grade drop out for everyone.

Verified 2026-09-11 in a logged-in browser: an Identify URL with an explicit `id=` list of five BC records showed exactly the four that still needed an ID; the fifth had reached Research Grade since the pull and was filtered out by Identify's default needs-ID filter, which is the intended behaviour.

## The lists

The page is rebuilt every day from the British Columbia records that need an ID and have a photo at that time, observed since 2025-01-01. Each record goes to one list by a keyed hash of its record id, so it stays on the same list in every daily build and the lists hold the same mix of taxa and observers on average. The key is kept privately and opened after the read-back. Each daily build serves the first 20 batches per list and taxon group, so the newest-first list stays new through the four weeks and records that were identified in the meantime drop out. A record served in the first 2 batches of its list on 3 builds in a row that still needs an ID moves behind every other record of its list for the next 7 days, then takes its normal place again. It is still served, only later. Without this, each list would put its hardest records in front of every participant every day, because a daily build has no memory of what it served and every participant starts each list at its first batch. The rule reads only the served log's record ids, batch positions and dates, so it treats every list the same.

- Newest first: batches are the newest records, as iNaturalist shows them today.
- Data-poor places first: batches are drawn from cells where iNaturalist has few or old records, using the where-to-blitz map scores. An ID there is often the only species record for that cell and goes straight into the models the lab uses to name 2027 field sites.
- Look-alikes together: batches are built so photos that look alike by an image model sit together, on the hypothesis that fewer context switches make identifying faster. The image model is BioCLIP 2.5, chosen on a 10,000-record BC sample where it placed same-species photos closer together than BioCLIP 2, DINOv2 and DINOv3 in every taxon group. It is weakest on fungi and fish (nearest-neighbour species agreement 43 and 53 percent, against 84 for amphibians), so look-alike batches in those groups will be looser.
- Unfamiliar photos first: each photo is compared by the same image model with every Research Grade BC photo of the same group, and batches are drawn from the records that look least like anything already verified. An ID there is the most likely to add a species, or a look of a species, that the verified record does not cover yet.

Every batch also carries two numbers in the build record, kept as information and never used to filter: how much its photos resemble each other, and how far they sit from verified photos of the same group. They are only available once the image embeddings are.

## What we measure, decided before the blitz

Only participants' identifications made during the blitz count. Each identification keeps its own timestamp, so this count is fixed when the blitz ends; it is read back within a week of the end. Record-level outcomes are read back 30 days after the end.

Each order is judged on the measure for its question. For each participant and list we count the records they gave a species-level ID:

- Data-poor places first: each record weighted by its where-to-blitz cell score, so an ID in a data-poor cell counts for more.
- Look-alikes together: the plain count.
- Unfamiliar photos first: the plain count. Species that reached Research Grade in a grid cell with no Research Grade record of that species before the blitz would suit this order better, but it is not in the code yet and is not tested.

For BC:

1. Primary: three comparisons with newest first, one per tested list, each on its own count fixed here: the weighted count for data-poor places first, the plain count for look-alikes together and for unfamiliar photos first. Per participant, the count on the list minus the same count on newest first. The test is a paired sign-flip permutation test on the sum of these differences, two-sided at 0.05, Holm-adjusted over these three primary comparisons only. Busy identifiers add the same effort to every list, so their skill cancels out. The analysis code pins these counts and stops on a list that has none.
2. Secondary, not Holm-adjusted: each list's other count (the plain count for data-poor places first, the weighted count for the other two), and a sign test on how many participants did better on each list.
3. Secondary: a record-level re-randomisation of the same summed difference. The sign-flip test re-randomises signs inside each participant, but the design randomises the record. This check holds each record's identifications fixed, draws each record's list again by the same keyed rule the build uses, and recomputes the summed difference, so it asks whether the result is unusual when only the split of records changes. It is secondary and does not replace the primary test.
4. Secondary, at 30 days: share of served records whose community taxon reached species, and Research Grade records added, weighted by cell score, per list.
5. Checks: the same test over the days before the blitz, when participants had not seen the page, should find nothing; each participant's reviewed records should split about evenly over the lists, which is what the rotation assumes; and for each list, the share of served records that someone other than a participant identified before any participant did, reported with its size and direction against newest first. Newest first is expected to lose more records this way, because other identifiers meet the newest records first, and that favours the tested lists. The observer's IDs on their own record, including the ID made at upload, are left out on both sides, so a record added during the blitz does not count as found first by someone else, and a participant's IDs on their own record do not count as a participant's. The read-back keeps the time of each ID but not the time a record reached Research Grade, so this check counts IDs only.

Expectation stated up front: newest first will likely win on the plain count, because records from remote cells have worse photos and are skipped more. Data-poor places first should win on the weighted count. If it loses on both, it is dropped.

Power is simulated by `python -m what_to_id.power_weighted` for two comparisons: data-poor places first against newest first on the weighted count, and one generic comparison on the plain count. The four lists enter only as the split of each participant's effort and the level: a run counts as a finding at Holm's first step, 0.05/3, so the power is a lower bound. A weighted run counts only when data-poor places first comes out ahead; a plain-count run counts a difference in either direction. No participant count is fixed; the runs cover 10 to 50 participants. The taxon group sizes are counts from the iNaturalist API on 2026-09-12, recorded in `power.py`. Data-poor places first on the weighted count: under the planned cap of 20 batches per list and taxon group, each of up to 120 records, which binds in 7 of the 10 taxon groups, the model gives the records participants meet on it an average map score of 0.361, against 0.158 on newest first. These averages come from the model, not from the pool. Its inputs are the preview's map scores: each list draws its scores with replacement from its group's 1,000 preview records, and data-poor places first keeps its top 2,400, so it serves scores like those of the top 16 of the 1,000 preview records in Plantae, the top 33 in Insecta and the top 44 in Fungi. The real pool's top 2,400 per list is not measured. The weighted count favours it while participants identify its records at more than 0.44 times their rate on newest first, so the weighted test mostly asks whether they slow down on data-poor records by more than that. At the same rate, power is 0.80 to 0.81 with 10 participants; at 0.8 times, 0.91 to 0.93 with 20; at 0.6 times, 0.31 to 0.38 with 20 and 0.79 with 50. The lead arises from the cap keeping each list's top scores, a mechanism of the model: in the preview, with no binding cap, both lists meet the same scores and data-poor places first comes out ahead in no more than 0.02 of runs, and the blitz model with no cap (`--max-batches 0`, 300 runs, 20 participants, factor 1.0, seed 0) gives 0.159 on newest first and 0.158 on data-poor places first, with no run ahead. On the plain count, the generic comparison that stands for look-alikes together and unfamiliar photos first, a lift of 0.3 gives 0.91 with 20 participants, 0.2 gives 0.78 to 0.81 with 25 and 0.99 with 40, and 0.1 gives 0.69 to 0.71 with 50. Ranges are two runs with different seeds.

The ID-rate factors, the lifts, the effort model (lognormal, median 100 records, sigma 1.5) and skill (Beta(2, 3)) are planning values with no source, because identifier effort from August 2026 is not available. The preview is the newest 1,000 records per taxon group, under a day of uploads for Plantae and about 82 days for Actinopterygii, not a random sample of the pool. The model draws map scores independent of record age and assumes one daily build, no repeat visits and one taxon group per participant. With no true difference, over 14,000 runs pooled over both seeds and all participant counts, the test found one in 0.025 of runs against a nominal 0.017, about 1.5 times as often and well beyond simulation error (about 0.001). The likely reason is that the split is drawn by record, and chance differences between lists make participants lean the same way, which the sign-flip test cannot see. That is why the record-level check is reported next to each comparison.

## What we are not claiming

- A method, tested first in BC. A result holds for this blitz's region and identifiers.
- No sorting by identification difficulty. That idea failed its pre-registered test and is not in this design.
- No wait-time numbers for the backlog. The pilot columns that produced them are censored.
- No value-of-information claims.
- No claim that batches change the order in which someone identifies, only which records they meet together.

## Reproducibility

Each daily build comes from a pull of the iNaturalist API at a recorded time, the private list key, and a pinned where-to-blitz commit whose grid is recorded by its hash in every build. Every day's served batches and record ids are logged by list letter, so exposure is fully known and the read-back is a join, not a reconstruction. The key maps letters to lists after the read-back. Four-list builds also record candidate and reference embedding fingerprints by taxon group. Replay requires matching files and checks the rebuilt served records against the saved log. The image model runs separately; daily builds consume a prepared, versioned embedding bundle.

## Open with the BC team

- Blitz dates.
- Which taxon groups identifiers will cover, if not all ten.
- The iNaturalist project participants join.

## Open before the blitz

- Prepare candidate and Research Grade reference embeddings, publish a verified bundle as versioned release assets, and configure its release and repository access. The daily script defaults to the four lists and rotation; it refuses to proceed without the required bundle. Configure the private LabelFirst dependency access, list key and blitz dates, then rehearse build and replay before enabling deployment. Preparation, publishing and deployment remain operational steps before launch.
