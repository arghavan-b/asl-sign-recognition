# ASL Sign Recognition — System Design

**Status:** Draft v2 · Phase 1 in implementation · continuous-signing readiness folded in (§10)
**Scope:** Landmark-based recognition of isolated ASL signs — MVP domain is everyday conversation; medical is deferred — designed to grow into a bidirectional sign ↔ spoken-language translation system.
**Audience:** Engineers, ML researchers, and clinical stakeholders working on the pilot.

---

## 1. Overview

This system recognizes American Sign Language signs from ordinary RGB video and converts them to text.

**MVP domain: everyday conversation.** The first deliverable targets high-frequency daily-conversation signs (greetings, question words, common verbs, feelings). These signs are well represented in public datasets like WLASL, so pretraining transfers cleanly, and the domain lets us validate the full pipeline — extraction, model, confidence gating, webcam demo — without the data-access, HIPAA, and recruitment overhead of a clinical setting. The **medical (ER/intake) domain is deliberately paused** for a later phase; that is where the gap between Deaf/Hard-of-Hearing (HoH) patients and clinical staff is most acute and human interpreters are scarce, but it carries deployment constraints best tackled once the core technology is proven. The medical vocabulary is preserved in `configs/medical_glosses.txt` for when that pilot resumes.

The guiding architectural decision — taken from the project roadmap — is to treat sign recognition as a **landmark sequence classification** problem rather than a raw-video computer-vision problem. MediaPipe Holistic extracts compact, signer-agnostic keypoints (hands, face, pose); a PyTorch sequence model classifies them. This is faster, far less data-hungry, and more robust across lighting and skin tone than training a video CNN from scratch.

The design is explicitly **phased**. Phase 1 (this document's primary focus) delivers an isolated-sign classifier with confidence scoring. Later phases add an interpreter-in-the-loop product layer, continuous (fluent) signing, and text→sign avatar output. The longer-term target architecture — bidirectional translation with confidence gating, hybrid edge-cloud compute, and avatar synthesis — is described in §11 so that Phase 1 choices do not foreclose it. §10 collects the Phase 1 decisions that specifically protect the continuous-signing goal.

### Design principles

The system prioritizes **accuracy and safety over raw speed**: in a clinical setting, a confidently wrong translation is worse than a flagged uncertain one, so confidence scoring is a first-class output from day one rather than a later bolt-on. It favors **compact representations** (landmarks, not pixels) to stay fast and generalizable. It is built to be **useful before the model is perfect** — the interpreter-in-the-loop loop (Phase 2) means the product has value while the AI is still improving, and every interpreter correction becomes a training example. Finally, it stays **narrow before broad**: nail medical ASL in one domain before expanding vocabulary, adding sign languages, or building avatar output.

---

## 2. Goals and non-goals

### Goals (Phase 1)

The Phase 1 system aims to correctly identify ~50–100 everyday-conversation signs at **≥80% accuracy on real users the model was not trained on**. It must run inference on a CPU or modest GPU in **≤300 ms per clip**, emit a calibrated **confidence score** with every prediction, and support both offline clip inference and a live webcam demo. It must cache landmark extraction so training does not re-run MediaPipe every epoch, and it must keep any video of real people off shared stores.

### Non-goals (Phase 1)

Phase 1 deliberately excludes continuous/fluent signing, text→sign avatar output, the audio (speech-to-text) pipeline, vision-language-model (VLM) recognition, AR/VR rendering, and multi-language support. The roadmap's analysis is that VLMs are not trained on sign data, are slow and expensive, and are outperformed on this task today by MediaPipe plus a trained sequence model; off-the-shelf Whisper already handles ASR; and AR/VR is product risk for an unvalidated market when a tablet exists in every hospital room. These are revisited in §11–§12.

---

## 3. System architecture (Phase 1)

```
  ┌─────────────┐   ┌────────────────────┐   ┌──────────────────┐
  │  RGB video  │──►│ MediaPipe Holistic │──►│ cached landmarks │
  │  clip (mp4) │   │   (extract.py)     │   │  (T, 1629) .npy  │
  └─────────────┘   └────────────────────┘   └─────────┬────────┘
                                                        │
                              ┌─────────────────────────▼──────────┐
                              │  PyTorch Dataset (dataset.py)       │
                              │  normalize · trim/pad · mask        │
                              └─────────────────────────┬──────────┘
                                                        │
                ┌───────────────────────────────────────▼───────────┐
                │  Sequence model (models/)                          │
                │   LSTM  ──or──  Transformer encoder                │
                └───────────────────────────────────────┬───────────┘
                                                        │
                         ┌──────────────────────────────▼────────────┐
                         │  softmax → label + confidence              │
                         │  conf < threshold → top-k candidates       │
                         │  (confidence gate, infer.py)               │
                         └────────────────────────────────────────────┘
```

The pipeline has a clean offline/online split. **Extraction** is an offline batch job run once over the corpus; its output (landmark arrays) is the durable artifact training consumes. **Training** reads cached arrays, never video. **Inference** re-uses the exact same extraction and normalization code path on a single clip or a sliding webcam buffer, guaranteeing train/serve consistency.

### Component responsibilities

`extract.py` decodes each video, runs MediaPipe Holistic frame by frame, flattens the four landmark groups into a fixed-width vector, and writes one `.npy` per clip mirroring the `data/raw/<gloss>/` layout. `landmarks.py` owns the feature specification, flattening, and normalization — the single source of truth for "what is a feature vector." `dataset.py` discovers labels from folder names, normalizes, fits each clip to a length budget, and collates variable-length clips into padded, masked batches. `models/` holds the two interchangeable classifiers behind a `build_model()` factory. `train.py` runs the loop, checkpoints the best model with labels baked in, and applies early stopping. `infer.py` loads a checkpoint and applies the confidence gate. `utils.py` centralizes config loading, seeding, and device selection.

---

## 4. Data

### Layout and labels

Raw clips live under `data/raw/<GLOSS>/<clip>.mp4`, where the folder name is the class label (gloss). Extraction mirrors this into `data/landmarks/<GLOSS>/<clip>.npy`. The MVP vocabulary is in `configs/daily_glosses.txt` — everyday signs (HELLO, THANK-YOU, WHERE, WANT, HAPPY, …) chosen for WLASL coverage. The deferred medical set lives in `configs/medical_glosses.txt` and should be **replaced and expanded from Phase 0 interviews** when the clinical pilot resumes.

### Sourcing strategy

Following the roadmap, the model is **pre-trained on [WLASL](https://github.com/dxli94/WLASL)** (a large public word-level ASL dataset) and then **fine-tuned on 5–10 signers recruited directly**, with even ~30 clips per sign giving a meaningful lift. WLASL and similar research datasets are acceptable for training a model used inside a pilot, but their **raw videos must not be redistributed** — only the trained weights and self-collected data leave the pipeline.

### The split that matters

The single most important evaluation discipline: **validation and test splits must be drawn from signers who do not appear in training.** Sign appearance varies enormously per person, so a random clip-level split inflates accuracy and hides the generalization failure that actually matters in deployment. The `data/splits/` hook supports pinning an explicit signer-disjoint split (`train.txt` / `val.txt` listing landmark paths). The current `train.py` falls back to a random split for bootstrapping only; a signer-disjoint loader is the first planned follow-up.

### Privacy and retention

Clinical footage stays on **encrypted local drives**. Once landmarks are extracted, raw video is **deleted unless the patient explicitly opts in** to longer storage. Landmarks are non-identifying relative to raw video, which both protects patients and shrinks what must be secured. This is a hard requirement, not a guideline — HIPAA exposure and patient trust both depend on it.

---

## 5. Landmark representation

MediaPipe Holistic returns four groups per frame: **pose** (33 points), **face** (468 points), **left hand** and **right hand** (21 points each). The system keeps `(x, y, z)` per point and drops pose visibility (noisy and weakly informative), yielding a default feature vector of:

```
pose 33×3 (99) + face 468×3 (1404) + left 21×3 (63) + right 21×3 (63) = 1629 dims/frame
```

Groups are toggleable in config; a hands-only variant is 126-d, useful for ablations and lightweight edge models. **Missing groups are encoded as zeros** (a hand out of frame, no face detected) so every frame has constant width and the model can learn the "absent" pattern rather than crashing on ragged input.

### Normalization

Each frame is **translation- and scale-normalized**: centered on the pose mid-hip (mean of the two hip landmarks) and divided by shoulder width. This makes the representation invariant to where the signer stands and how far they are from the camera — two confounds that otherwise leak into the model and hurt cross-signer generalization. When pose is unavailable the normalization degrades gracefully to a no-op. Normalization is applied identically at train and inference time (it lives in `landmarks.py`, called by both `dataset.py` and `infer.py`).

### Feature evolution and spec versioning

Three changes are planned for continuous readiness. The face block shrinks from all 468 points to a **lips-and-eyebrows subset** (~50 points): the full face is 1404 of the 1629 dims and mostly redundant, while lips and brows carry the grammatical non-manual signal (§10). Per-frame **velocity (delta) features** are added, since motion dynamics materially help sign segmentation. And training data is **mirror-augmented** so left-handed signers are covered essentially for free.

Because the cached landmark corpus silently breaks whenever the feature spec or normalization changes, the spec is **versioned**: a hash of the spec is embedded in cache metadata so stale caches are detected rather than silently trained on. Relatedly, MediaPipe Holistic's legacy "solutions" API is deprecated — migration to the MediaPipe Tasks Holistic Landmarker should happen **before** a large corpus is extracted, to avoid a full re-extraction later.

### Temporal fidelity

Extraction stores per-clip fps and timestamps, and sequences are resampled to a **fixed frame rate** rather than uniformly subsampled to a frame budget. Uniform subsampling (the current `_prep` behavior) destroys signing-speed information — a key sign-boundary cue — and webcam fps varies at runtime, so a rate-normalized timeline is required for train/serve consistency in the continuous setting.

---

## 6. Model design

Two architectures are provided behind one factory, switchable via `model.type` in config.

**BiLSTM (`models/lstm.py`).** An input projection feeds a multi-layer bidirectional LSTM; outputs are masked-mean-pooled over valid timesteps and classified by a LayerNorm→Dropout→Linear head. `pack_padded_sequence` ensures padding never contributes to the recurrence. Strong baseline, cheap to train, good with limited data.

**Transformer encoder (`models/transformer.py`, default).** An input projection, a learned `[CLS]` token, sinusoidal positional encoding, and a stack of `TransformerEncoderLayer`s (GELU, pre-configured heads/depth). A `src_key_padding_mask` excludes padded frames from attention; the `[CLS]` representation is classified. Scales better with more data and captures long-range co-articulation structure that matters once continuous signing arrives.

Both consume `(B, T, F)` plus a boolean padding mask `(B, T)` and emit logits `(B, C)`. Keeping the interfaces identical lets the rest of the system stay model-agnostic and makes head-to-head comparison a one-line config change. Default capacity (hidden 256, 4 layers, 8 heads) is sized for a ~100-class problem with thousands of clips; expect to tune down for tiny fine-tuning sets and up as data grows.

Both heads currently emit a single clip-level prediction (pooled output or `[CLS]`), which discards exactly the per-frame representations a CTC/seq2seq decoder needs. To keep Phase 3 a head-swap rather than a retrain, the encoders **retain frame-level outputs** and an **auxiliary frame-level head** is planned. Alongside it, an explicit **blank/transition class** is trained from negative samples (rest poses, inter-sign transitions, idle motion): continuous signing is mostly movement epenthesis between signs, and a model that has never seen "not a sign" cannot find boundaries — this is the cheapest single unlock for segmentation, and it is trainable with isolated-sign data today.

---

## 7. Training

Training uses **AdamW** with weight decay, a **cosine-annealing** learning-rate schedule, gradient clipping, and **early stopping** on validation accuracy. The best checkpoint is saved with the full config and label list embedded, so inference is fully self-describing — `infer.py` needs only the `.pt` file. Cross-entropy is the loss; class imbalance (some signs will have far fewer clips) should be addressed with class weights or sampling once real data lands.

Reproducibility is handled by a single global seed across Python, NumPy, and Torch. Device selection is automatic (CUDA → Apple MPS → CPU) but overridable. The recommended regimen follows the roadmap: pre-train on WLASL for many epochs until dev WER/accuracy plateaus, then continue fine-tuning on the self-collected medical clips. After Phase 2 is live, retrain weekly on the latest interpreter-corrected transcripts and push updated weights. Weekly retraining without guardrails will eventually ship a regression, so **model and data versioning, a pinned signer-disjoint regression eval set, and a rollback path** are part of the training design from the start, not an ops afterthought.

---

## 8. Evaluation

The headline metric is **top-1 accuracy on unseen signers**; the roadmap rightly notes this number "will be humbling but essential." Secondary metrics include top-3 accuracy (relevant because the product surfaces candidates when uncertain), per-class accuracy (to find which signs need more data), and a confusion matrix (to find systematically conflated sign pairs, e.g. PAIN/HURT). Confidence **calibration** matters as much as accuracy: the gate in §9 is only meaningful if a reported 0.8 actually corresponds to ~80% correctness, so reliability diagrams and post-hoc temperature scaling are part of evaluation, not an afterthought. Latency is benchmarked against the ≤300 ms/clip budget on target hardware (restated per-frame once streaming inference lands, §10).

For continuous readiness, **WER on a continuous benchmark** (How2Sign-style) is tracked from the first sequence-to-sequence prototype — scores will be poor at first, but they make the Phase 3 cliff measurable rather than rhetorical. Translation output is additionally judged by **comprehension testing with Deaf raters** plus a semantic similarity metric, since WER alone penalizes meaning-preserving paraphrase and misses meaning-breaking word swaps.

---

## 9. Inference and confidence gating

`infer.py` runs the identical extraction + normalization path on either a stored clip or a live webcam buffer (a sliding window of the most recent frames). It produces a softmax distribution and applies the **confidence gate**: if the top probability is at or above the threshold (default 0.8), it returns a single confident prediction; below it, it returns the **top-k candidates** flagged for clarification. This is the seam where Phase 2 attaches — instead of guessing, the system can ask the user to repeat or fingerspell, present "Did you mean PAIN, HURT, or NAUSEA?", or route the clip to a remote interpreter. Building confidence output now means that product layer requires no model rework later. The webcam demo color-codes the overlay (green confident, amber gated) to make the gate's behavior legible during development.

The sliding-window demo is a development tool, not the continuous architecture: streaming inference, signing-activity detection, and the hypothesis commit policy are specified in §10, and the gate's *semantics* (confident vs. flagged) survive that transition even though its implementation (per-clip softmax) does not.

---

## 10. Continuous-signing readiness

> The full Phase 3 system design lives in `docs/CONTINUOUS_DESIGN.md`; this section covers only the Phase 1 decisions that protect it.

Phase 3 is where the research risk lives, so Phase 1 must avoid foreclosing it. This section fixes the decisions that are cheap now and expensive later, plus the streaming behaviors that must be designed before continuous decoding exists. (Frame-level outputs, the blank/transition class, temporal fidelity, and feature evolution are covered in §5–§6.)

**Streaming inference.** Re-classifying a full sliding window every step is wasteful and produces flickery output. The continuous path targets a temporal-convolution front-end with **causal/chunked attention**, with the latency budget restated **per-frame** rather than per-clip. A cheap **signing-activity detector** (the VAD equivalent of speech systems) gates the full model: it saves compute, prevents hallucinated output from idle video, and is the natural on/off switch for an always-on camera (§14).

**Output policy: partial vs. final hypotheses.** Like streaming ASR, continuous decoding revises its hypothesis mid-sign. The design needs an explicit **commit policy** — when text is first shown, when it may be revised, and when it freezes — because revision behavior drives the product feel more than raw accuracy does. **Utterance boundaries** are largely non-manual (pauses, head/brow position, hands returning to rest) and determine when a completed sequence is handed to translation.

**Decoding stack and the gloss decision.** The default plan: CTC posteriors → beam search with a gloss language model → LLM for gloss→English (handling Topic-Comment → Subject-Verb-Object). The confidence gate moves from per-clip softmax to **per-segment CTC posterior/entropy**. One decision is held open deliberately (§15): recent sign-translation work is increasingly **gloss-free** (landmarks → English directly), because gloss annotation is the data bottleneck and glosses lose information — classifiers, depicting signs, and role shift do not gloss cleanly. The gloss interface is the default, not a commitment.

**Linguistic coverage.** **Non-manual grammar** (questions, negation, intensity) lives in the face and head, not the hands; a gloss sequence alone yields wrong English ("YOU GO" vs. "Are you going?"). It is modeled either as auxiliary classifier heads or absorbed end-to-end — but it must be modeled. **Fingerspelling** (names, out-of-vocabulary words) is unavoidable in continuous ASL and is handled by a dedicated recognizer (trained on ChicagoFSWild-class data) behind a fingerspelling detector, not as one gloss class per letter.

**Data for continuous.** Pretraining extends beyond WLASL to continuous corpora — **YouTube-ASL** (~1,000 h), **OpenASL**, and **How2Sign** — and synthetic continuous data is generated by concatenating isolated clips with interpolated transitions, a known bootstrap for CTC training.

**Per-signer enrollment.** A ~30-second few-shot adaptation pass (calibrating to one signer's style) often buys more than weeks of general training; the serving layer reserves a hook for it.

**Pipeline concurrency.** Extraction, model inference, and decoding run at different rates. The runtime is designed as queued stages with an explicit **backpressure policy** — what drops, and what the user sees, when the model falls behind — rather than today's synchronous loop.

---

## 11. Target architecture (later phases)

The long-term system is **bidirectional**: sign→text/speech and text/speech→sign, with confidence gating at each stage. Its eventual shape (per the project's patent direction) includes a VLM/landmark front-end for recognition, an LLM for grammatically-aware translation between sign structure (Topic-Comment) and spoken structure (Subject-Verb-Object), parametric **avatar synthesis** for sign output, and a **hybrid edge-cloud** split that runs lightweight preprocessing and rendering on-device while heavier inference runs in the cloud, adapting to network and device conditions. AR/VR rendering and multi-language support sit at the far end.

Phase 1 is forward-compatible with this: landmark features are exactly the signer-agnostic representation the recognition front-end needs; confidence gating is already a first-class concept; and the clean extract/model/infer separation means the recognition stage can later be swapped or ensembled without touching the data or serving layers.

---

## 12. Phased roadmap

**Phase 0 — Medical discovery (deferred, paired with the medical pilot).** When the medical domain resumes: interview ≥10 Deaf/HoH patients and ≥5 interpreters to fix the 50–100 signs covering ~80% of ER/intake conversations, locate where the interpreter bottleneck actually hurts, and learn deployment constraints (HIPAA BAA, EHR integration, procurement). For the everyday-conversation MVP, vocabulary selection is lighter-weight — start from `configs/daily_glosses.txt` and refine with a few Deaf signers.

**Phase 1 — Isolated sign → text (months 1–3).** The system in this document: a ~50–100-sign **everyday-conversation** classifier at ≥80% accuracy on unseen signers, confidence scoring, clip + webcam inference. Medical vocabulary is paused and slots back in (as a fine-tuning domain) once the pipeline is proven.

**Phase 2 — Confidence gating + interpreter-in-the-loop (months 3–5).** The actual differentiator. Surface candidates or route low-confidence clips to a remote interpreter via a thin console; every correction is a labeled example. Delivers a useful product before the AI is perfect, and reveals the right operating threshold.

**Phase 3 — Continuous signing (months 5–9).** The real research challenge: fluent signing breaks isolated-sign accuracy through co-articulation, dropped hands, and speed variation. Move to a sequence-to-sequence model (frame landmarks → CTC → gloss sequence → text), pretraining on YouTube-ASL/OpenASL and benchmarking against PHOENIX-2014 and How2Sign (see §10 for the full readiness plan, including the gloss-free alternative). The Phase 2 correction loop is the data-collection engine. Expect accuracy to drop; set expectations internally.

**Phase 4 — Text → sign avatar (parallel, from ~month 4–5).** Start with **video-clip retrieval** (a dictionary of high-quality recorded medical phrases with transition blending), comprehension-tested by Deaf users, before investing in parametric/procedural synthesis. Naturalness is the hard bar; budget 6+ months for avatar quality.

**Phase 5 — Expand (month 9+).** Only after a validated, hospital-deployed medical pipeline: broaden vocabulary, add a second sign language, push inference to edge devices, and build AR overlays *if* validated hospital demand exists.

### Honest timeline

| Milestone | Realistic timeframe |
| --- | --- |
| Everyday-conversation classifier demo (~50–100 signs) | 2–3 months |
| Re-target / fine-tune to medical vocabulary | +1–2 months |
| Interpreter-in-the-loop pilot at one hospital | 5–6 months |
| Continuous-signing prototype | 9–12 months |
| Commercially viable product | 18–24 months |

The core strategic insight: **the confidence gate + interpreter loop is the product for the first ~12 months.** The AI improves in the background, fed by real clinical sessions — a defensible wedge, where most teams try to perfect the AI first and never ship.

---

## 13. Risks and mitigations

The dominant technical risk is **poor cross-signer generalization** — mitigated by signer-disjoint evaluation, landmark normalization, and WLASL pre-training, and surfaced early rather than hidden by a random split. **Confidence miscalibration** would undermine the gate that the whole safety story rests on; mitigated by calibration metrics and temperature scaling. **Data scarcity** in the medical domain is mitigated by pre-training plus the Phase 2 correction loop turning usage into labels. The **continuous-signing cliff** (Phase 3) is a known step-change in difficulty, scheduled accordingly rather than assumed away. On the product side, the chief risk is **over-building** — VLMs, avatars, AR/VR, multi-language — ahead of validation; the phased plan and explicit non-goals exist precisely to resist it. **MediaPipe failure modes** (occlusion, motion blur, poor lighting, two signers in frame) degrade features silently; the zero-fill convention keeps the system running, and input-quality metrics should feed the recognition confidence score. **Camera geometry** is a related deployment risk: seated or bed-bound signers and off-axis tablet placement present viewpoints the normalization does not remove; mitigations are 3D rotation augmentation of landmarks or constrained device placement, decided once early pilot footage shows which viewpoints actually occur. Finally, **silent model regressions** from the weekly retrain cadence are mitigated by the pinned regression eval set, versioning, and rollback path specified in §7.

---

## 14. Ethics, licensing, and compliance

Research datasets (WLASL, How2Sign) may train models shipped into a commercial pilot, but their **raw videos may not be redistributed**. Avatar clips generated by the team are owned by the team, subject to model-code licenses (e.g. GenASL is Apache-2.0). All clinical footage stays **encrypted and local**, with raw video deleted post-extraction unless the patient opts in. Continuous mode changes the privacy posture from "record a clip" to an **always-on camera**: consent language must reflect that, the signing-activity detector (§10) doubles as the hard on/off switch, and **on-device landmark extraction** — only landmarks ever leave the device, never video — becomes a design requirement rather than an optimization. Deployment in a US hospital requires a **HIPAA Business Associate Agreement** and likely EHR-integration and procurement review — items to confirm during Phase 0, not after building. Throughout, Deaf consultants should be involved in vocabulary selection, avatar comprehension testing, and evaluation, both as an accuracy measure and as a matter of building *for* the community rather than merely *about* it.

---

## 15. Open questions

Several decisions are deliberately deferred until Phase 0 data and early experiments land: the exact medical vocabulary and its size; the right confidence threshold (to be learned from the interpreter loop, not guessed); whether LSTM or Transformer wins at the available data scale; how much WLASL pre-training transfers to the medical sub-domain; the minimum clips-per-sign for acceptable fine-tuning; and the precise edge-cloud partition once latency and connectivity constraints are measured in a real clinical environment. Continuous-signing readiness (§10) adds its own deferred decisions: whether Phase 3 keeps the gloss interface or goes gloss-free; the hypothesis commit/revision policy for streaming output; camera-placement constraints versus viewpoint augmentation; and the shape of per-signer enrollment.
