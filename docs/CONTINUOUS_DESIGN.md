# Continuous ASL → Text — System Design (Phase 3)

**Status:** Draft v1
**Scope:** Streaming recognition and translation of fluent, continuous ASL into English text. Companion to `DESIGN.md` (Phase 1 isolated-sign system); assumes its landmark pipeline, privacy posture, and confidence-gating philosophy.
**Audience:** Engineers and ML researchers building Phase 3.

---

## 1. Problem statement

Isolated-sign classification answers "which one of N signs is this clip?" Continuous signing breaks that framing in four ways. **Co-articulation:** each sign's handshape and trajectory is deformed by its neighbors, so dictionary citation forms are systematically wrong as templates. **No boundaries:** signs flow through transition movements (movement epenthesis); segmentation and recognition must happen jointly. **Prosody and reduction:** fluent signers speed up, reduce movements, and drop the non-dominant hand. **Output mismatch:** the target is English text, not a gloss-per-clip — ASL grammar (Topic-Comment, spatial referencing, non-manual marking) must be *translated*, not transliterated.

Formally the system is two coupled tasks: **CSLR** (continuous sign language recognition: landmarks → gloss sequence, measured by WER) and **SLT** (sign language translation: landmarks → English, measured by BLEU/BLEURT and human comprehension). We build both — CSLR as a supervision scaffold and diagnostic, SLT as the product output — and keep the gloss interface replaceable (§3, sauce S6).

---

## 2. Goals and non-goals

**Goals.** Streaming ASL→English on a tablet-class device + cloud, with: per-segment confidence gating compatible with the Phase 2 interpreter loop; partial hypotheses on screen within ~1–2 s of signing; vocabulary beyond the Phase 1 gloss set via fingerspelling and gloss-free translation; signer-disjoint evaluation throughout.

**Non-goals.** Sign output (avatar), multi-signer scenes (one active signer assumed, §5), non-ASL sign languages, offline-only edge inference for the full model (the edge runs extraction and the lightweight student, §5).

---

## 3. Architecture

```
            ┌────────────── edge (device) ──────────────┐  ┌───────── cloud ─────────┐
 camera ──► │ Holistic landmarks (Tasks API, fixed fps)  │  │                          │
            │   │                                        │  │  ┌────────────────────┐  │
            │   ├─► signing-activity detector (tiny GRU) │  │  │ streaming Conformer │  │
            │   │      gates everything downstream       │──┼─►│ encoder (chunked    │  │
            │   └─► stream split: hands · pose · face    │  │  │ causal attention)   │  │
            │        + Δ/ΔΔ features, normalization      │  │  └──────┬─────────────┘  │
            └────────────────────────────────────────────┘  │         │ frame states   │
                                                            │   ┌─────┴──────┬───────────────┐
                                                            │   ▼            ▼               ▼
                                                            │ CTC gloss   fingerspell    non-manual
                                                            │ head + LM   CTC head       heads (Q/neg)
                                                            │   └──────┬─────┴───────────────┘
                                                            │          ▼
                                                            │  projector → LLM decoder (LoRA)
                                                            │  → English partial/final hypotheses
                                                            │  → per-segment confidence gate
                                                            └──────────────────────────────────┘
```

The encoder is shared; heads are cheap. Everything downstream of landmarks consumes the same `(T, F)` representation as Phase 1 — `landmarks.py` remains the single source of truth, extended with stream splitting and delta features.

---

## 4. The secret sauces

Each item below is a known, validated technique from speech recognition or recent SLT literature, adapted to this system. Together they are the difference between a demo that works on the bench and a system that survives a real signer.

### S1 — Multi-stream encoding with late fusion

Don't feed one 1629-d vector into one encoder. Split into three streams — **hands** (2×21×3 + deltas), **pose/body** (upper-body subset), **face** (lips + eyebrows subset) — each with its own small input encoder, fused by cross-attention into the shared Conformer trunk. Hands carry lexical identity; pose carries spatial grammar and sign boundaries; face carries grammatical marking. Separate streams let each be normalized, dropped out, and pretrained independently, and ablate cleanly. This is the TwoStream-SLR insight, extended to three streams; it consistently beats monolithic input on PHOENIX-class benchmarks.

### S2 — Self-supervised pretraining on unlabeled signing

The cheapest hours are unlabeled ones. Before any supervised training, pretrain the encoder with **masked landmark modeling**: mask random temporal spans and one random stream, predict the masked landmarks (or a quantized codebook of hand poses) from context. Run this over *everything* — YouTube-ASL video (ignore captions), How2Sign, our own recordings. This is the SignBERT/BEST/SSVP-SLT recipe and the wav2vec2 lesson from speech: SSL pretraining is worth more than any architecture tweak when labeled data is scarce. The quantized hand-pose codebook doubles as an interpretable "phonetic" inventory of ASL handshapes.

### S3 — Synthetic continuous data from isolated clips

We own a labeled isolated corpus (Phase 1). Manufacture continuous training data from it: sample gloss sequences (from ASL corpus n-grams or LLM-generated plausible sentences), concatenate the corresponding isolated clips, and **synthesize movement epenthesis** between them by interpolating landmarks from one sign's end pose to the next sign's start pose with velocity-profile smoothing and randomized transition duration. Add co-articulation simulation: blend a sign's first/last few frames toward its neighbor. The result is unlimited CTC training data with perfect gloss alignments — wrong in detail, right in structure. Train on synthetic first; real continuous data then fine-tunes away the artifacts. This bootstrap is what makes CTC converge before any real aligned data exists.

### S4 — Blank/transition as a first-class class + activity gating

CTC's blank symbol will absorb transitions, but help it: explicitly train a **transition/no-sign class** from negative samples (rest poses, epenthesis segments from S3 synthesis, idle motion, conversational gesticulation that isn't ASL). Upstream, a **tiny signing-activity detector** (2-layer GRU on hands+pose, <1 ms/frame) runs always-on at the edge and gates the entire cloud pipeline — saving compute, preventing hallucinated output from idle video, and serving as the privacy on/off switch. Speech systems call this VAD; no streaming system ships without it.

### S5 — Joint CTC + translation training (the hybrid loss)

Train the encoder with **CTC gloss loss and translation loss simultaneously** (weighted sum, CTC weight annealed down over training). CTC provides monotonic alignment pressure that pure attention lacks — it forces frame states to localize sign identity in time, which is exactly what streaming, segmentation, confidence attribution, and the interpreter loop all need. Where real gloss annotations don't exist (YouTube-ASL has only captions), generate **pseudo-glosses**: lemmatize and reorder caption words with an LLM into plausible gloss sequences (recent work shows this closes most of the gloss-free gap). The CTC head is scaffold as much as product: even if the shipped output is the LLM's, CTC posteriors drive segmentation and gating.

### S6 — LLM as the translation decoder

Don't train a translation decoder from scratch — **project encoder states into a pretrained LLM** (small enough to serve: 1–8 B class, LoRA-tuned, frozen base). The projector is a light adapter (linear or Q-Former-style) mapping pooled sign segments to LLM token-embedding space; the LLM consumes a prompt (domain context, conversation history, candidate glosses from the CTC head) plus the sign embeddings and emits English. This is the Sign2GPT/SpaMo/MMSLT direction, current SOTA in gloss-free SLT. Benefits: the LLM brings world knowledge and fluent English for free, handles Topic-Comment→SVO restructuring, exploits **dialogue context** (previous turns disambiguate one-handed reduced signs), and integrates fingerspelled fragments and candidate lists natively. The gloss interface (S5) stays as scaffold, so this decoder is swappable — gloss-based fallback if the LLM path underperforms.

### S7 — Streaming via chunked attention + local-agreement commit

The encoder is a **Conformer with chunked causal attention**: frames attend within a chunk (~0.6–1 s) plus a bounded left context; right context is limited to one chunk of lookahead. This bounds latency structurally rather than by re-running an offline model on sliding windows (which is both wasteful and flickery). For output stability, use the **local agreement policy** from simultaneous translation: a partial hypothesis is shown immediately (grey), but text is *committed* (black, frozen) only when two consecutive decoding steps agree on it. This single rule converts a flickering decoder into a usable live-captioning UX, and its agreement horizon is the one knob trading latency against revision rate.

### S8 — Offline teacher, streaming student

Train the best possible **offline, bidirectional, full-context model** first — no latency constraints, full attention, test-time augmentation. Then **distill** it into the streaming model: the student matches the teacher's frame posteriors and translation outputs on unlabeled data. The teacher also back-labels the data engine (S12) with high-quality pseudo-labels. Never make the streaming model learn the task from scratch; make it imitate a model that already solved it. Standard practice in production ASR; rarely done in SLT — a real edge.

### S9 — Fingerspelling subsystem

Continuous ASL is ~12–35% fingerspelling in some registers (names, brands, OOV words) — ignoring it caps usefulness regardless of sign-vocabulary size. Run a **fingerspelling detector** (binary head on the shared encoder) and, when active, a dedicated **letter-CTC recognizer** on a hands-only, higher-frame-rate stream (fingerspelling is fast — 5–8 letters/s; this is where fixed-fps extraction pays off). Train on ChicagoFSWild(+). Decode letters with a lexicon + character LM, and pass the result to the LLM decoder as a text fragment ("fingerspelled: M-A-R-I-A") rather than forcing it through the gloss vocabulary.

### S10 — The non-manual channel as explicit supervision

Questions, negation, conditionals, intensity, and role shift are marked on the face and head. Add **auxiliary heads** on the face stream predicting: question type (yes/no = raised brows; wh- = furrowed), negation (headshake span), and mouthing presence. Even with coarse labels (weakly supervised from captions: caption ends in "?" → question marker somewhere in the segment) these heads force the face stream to learn grammatical features, and their outputs feed the LLM prompt ("the signer marked this as a question"). This is the difference between transcribing "YOU GO STORE" and producing "Are you going to the store?".

### S11 — Signer invariance + 30-second enrollment

Two complementary moves. At training time, **adversarial signer invariance**: a signer-classifier head on the encoder trained with gradient reversal, penalizing the encoder for encoding signer identity. At inference time, **enrollment**: a ~30-second calibration (user signs a known passage) used for (a) per-signer landmark statistics (body proportions beyond what normalization removes, habitual signing-space size) and (b) optional test-time adaptation of input-projection layers (BitFit-scale, seconds of compute). Speech solved speaker variance with exactly this pairing (i-vectors → adaptation); per-signer WER drops are typically the largest single-user win available.

### S12 — The data engine (the moat)

The model is temporary; the flywheel is the asset. Every production interaction generates training signal: **interpreter corrections** from the Phase 2 loop (gold continuous labels, the scarcest resource in the field), **user confirmations** ("did you mean X?" → accepted = weak positive), and **teacher back-labeling** (S8) of all incoming unlabeled footage. Weekly retrains consume this; the pinned signer-disjoint regression set and rollback path (DESIGN.md §7) guard it. Prioritize labeling by **uncertainty × frequency**: segments where the model is unsure *and* the construction is common get human attention first. Within a year this corpus exceeds every public continuous-ASL dataset in domain relevance — that, not the architecture, is the defensible asset.

### S13 — Augmentation suite (landmark-space, nearly free)

All in landmark space, all cheap: **temporal warping** (random speed 0.7–1.4×, per-segment), **mirroring** (left/right hand swap — covers left-handed signers), **3D rotation** (±15–20° yaw/pitch about the body center — covers off-axis camera placement, the bed-bound/tablet geometry risk), **landmark jitter and dropout** (simulates MediaPipe noise and occlusion — drop a hand for random spans), **spatial scale/anisotropy** (body-shape variation), and **frame drops** (simulates real webcam delivery). Phase 1 trained without most of these; Phase 3 cannot.

### S14 — Per-segment confidence and gating

The Phase 1 gate was one softmax per clip; the streaming gate is **per-segment**: CTC posterior probability over the segment's most-likely gloss span, sequence-level entropy, agreement between the CTC path and the LLM output (when they disagree, something is wrong), and input-quality signals (landmark dropout rate, activity-detector confidence). Calibrate with temperature scaling on a held-out signer-disjoint set; verify with reliability diagrams per segment length. Low-confidence segments render amber in the UI, trigger clarification ("did you mean…?"), or route to the interpreter — unchanged Phase 2 semantics, new mechanics.

---

## 5. Serving architecture

**Pipeline stages** (each an async queue-connected worker): capture → landmark extraction → activity gate → feature assembly → encoder → decoders → commit policy → UI. Explicit **backpressure policy**: if the encoder falls behind, drop frames at the feature-assembly stage uniformly (never burst-drop), surface a "catching up" indicator past 1 s of lag, and never block capture.

**Edge/cloud split.** Edge (tablet): camera, landmark extraction (MediaPipe Tasks, GPU-delegated), activity detector, feature assembly, UI. Cloud: encoder, heads, LLM. Only landmarks cross the wire (~80 KB/s at 30 fps float16 — fine on hospital Wi-Fi; and the privacy requirement of DESIGN.md §14 is met structurally: video never leaves the device). A **distilled student** (S8, quantized, hands+pose only) ships on-device as a degraded-connectivity fallback delivering gloss-level output without the LLM.

**Latency budget** (frame arrival → committed text): extraction ~15 ms · activity gate ~1 ms · network ~30 ms · encoder chunk wait ~400 ms avg + compute ~40 ms · LLM partial ~150 ms · commit-policy delay ~1 chunk. Target: **partial text < 1 s, committed text < 2.5 s** behind the signing. These are budgets to benchmark against, not estimates to trust.

---

## 6. Evaluation

Four layers, all signer-disjoint. **CSLR:** WER against gloss references (own corpus + PHOENIX-2014 for sanity, How2Sign where gloss exists). **SLT:** BLEU and BLEURT (or COMET-style learned metric) against English references on How2Sign and held-out interpreter-corrected sessions; learned metrics weigh meaning over n-gram overlap and matter more than BLEU here. **Streaming:** average lagging (AL) for latency; **erasure rate** (committed-then-revised characters — must be ~0 by construction; partial-revision rate is the tunable) for stability. **Human:** comprehension testing with Deaf raters on real conversations — the only metric that ultimately counts — plus interpreter-correction rate in production, which doubles as a free longitudinal metric from the data engine.

Calibration (reliability diagrams, ECE) is evaluated per segment, per the gate's needs. Every release must beat the pinned regression set before deploy (DESIGN.md §7).

---

## 7. Data plan

| Source | Hours | Supervision | Role | License note |
| --- | --- | --- | --- | --- |
| YouTube-ASL | ~980 (shrinking) | English captions | SSL pretrain + pseudo-gloss SLT | IDs CC BY 4.0; videos under YouTube ToS — assess for commercial training |
| How2Sign | ~80 | Sentence-aligned English | SLT fine-tune + benchmark | research-friendly; verify for commercial use |
| OpenASL | ~288 | English captions | ⚠ CC BY-NC-ND — **excluded from commercial training**; research comparisons only |
| Synthetic (S3) | unlimited | Perfect gloss alignment | CTC bootstrap | self-owned |
| Phase 1 isolated corpus | grows | Gloss labels | S3 source + isolated pretrain | self-owned |
| ChicagoFSWild(+) | ~7k sequences | Letter labels | Fingerspelling (S9) | verify license |
| Interpreter corrections (S12) | grows weekly | Gold continuous | Fine-tune + eval | self-owned, consented |

Acquisition: YouTube sources are ID lists — download via `yt-dlp`, extract landmarks immediately, retain only landmarks (storage and privacy both). Expect dataset decay; snapshot the landmark corpus, version it (spec hash, DESIGN.md §5).

---

## 8. Training recipe (order matters)

1. **SSL pretrain** encoder on all landmark data, masked landmark modeling (S2). Longest stage; do once per spec version, reuse forever.
2. **Isolated supervised** fine-tune with the Phase 1 corpus + frame-level auxiliary head + blank class (S4) + full augmentation (S13).
3. **Synthetic continuous CTC** (S3): converge CTC on stitched data; verify alignments visually on held-out synthetic.
4. **Real continuous, weak supervision:** YouTube-ASL/How2Sign with pseudo-gloss CTC + translation loss (S5), adversarial signer invariance on (S11).
5. **LLM alignment** (S6): freeze encoder mostly, train projector + LoRA on sentence-aligned pairs; add dialogue-context training (previous-turn conditioning).
6. **Joint polish:** unfreeze, low LR, all losses, CTC weight annealed; add fingerspelling and non-manual heads (S9, S10).
7. **Distill to streaming student** (S8); calibrate (S14); quantize edge fallback.
8. **Production loop:** weekly fine-tunes from the data engine (S12) re-running stages 6–7 incrementally, gated by the regression set.

Skipping straight to stage 4 (the obvious temptation) is how continuous-signing projects die: without S2's representations and S3's alignment bootstrap, CTC on weak labels collapses to the blank symbol and attention learns caption priors instead of signing.

---

## 9. Risks

**Pseudo-gloss quality ceiling** — pseudo-glosses inherit LLM word-order guesses; mitigated by the data engine replacing them with interpreter gold over time, and by the gloss-free path not depending on them. **LLM hallucination** — fluent-but-wrong English is the worst failure mode for a medical setting; mitigated by CTC/LLM agreement gating (S14), conservative decoding (low temperature, candidate-constrained prompts), and the interpreter loop owning low-confidence output. **Latency creep** — every component is "only 50 ms"; owned by a single end-to-end latency budget with per-stage CI benchmarks. **Dataset decay** (YouTube deletions) — snapshot landmarks now. **Streaming/offline quality gap** — if distillation can't close it, fall back to segment-batched decoding (commit per utterance instead of per chunk; worse latency, same quality). **Left-field risk:** MediaPipe landmark quality is the floor for everything; track its failure rate as a first-class metric and budget for swapping the front-end (the multi-stream interface makes the extractor replaceable).

---

## 10. Open questions

Chunk size vs. user-perceived latency (needs user testing, not benchmarks); LLM size/placement (1 B on-device vs. 8 B cloud — decided by the hallucination-vs-latency tradeoff measured in stage 5); whether the CTC scaffold can be dropped post-convergence or remains permanently load-bearing; pseudo-gloss generation recipe (lemmatize-and-reorder vs. LLM freeform); how much dialogue context helps vs. leaks errors forward; enrollment UX (explicit passage vs. silent accumulation over the first minutes).
