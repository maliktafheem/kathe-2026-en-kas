# KATHE 2026 — English to Kashmiri translation

Team **IT**. Public leaderboard score **13.90** on the KATHE 2026 metric,
`sqrt(BLEU × chrF++)`.

The model is a fine-tune of `ai4bharat/indictrans2-en-indic-1B` (1.1B
parameters) that translates English (`eng_Latn`) into Kashmiri in the
Perso-Arabic script (`kas_Arab`).

| | |
|---|---|
| Weights | [`maliktafheem/kathe-2026-en-kas-arab-1b`](https://huggingface.co/maliktafheem/kathe-2026-en-kas-arab-1b) |
| Base model | `ai4bharat/indictrans2-en-indic-1B` |
| Score | `13.90` public leaderboard |

## Install

```bash
pip install -r requirements.txt
```

`transformers` must stay on 4.51.x. Version 5 removed
`tokenizer.as_target_tokenizer()`, which this architecture needs in order to
decode output ids in target-vocabulary space.

The weights download from the Hugging Face Hub on first use, about 4.5 GB. A
GPU is not required. On CPU the model runs in float32 and needs about 5 GB of
RAM.

## Run

Check that the weights load and translate:

```bash
python load_model.py
```

One sentence:

```bash
python infer_single.py "The sun rises in the east."
```

A whole file. The input needs an `ID` column and a `sentence` column. The
output has `ID` and `kashmiri_text`, in the same row order, which is the
competition submission format:

```bash
python infer_batch.py --input examples/sample_input.csv --output predictions.csv
```

Both scripts accept `--model` to point at a local directory instead of the Hub,
and `--device` to force `cuda` or `cpu`.

## Methodology

### The finding that produced the score

`IndicProcessor.preprocess_batch(..., is_target=True)`, from the standard
IndicTrans2 toolkit, deletes three Arabic short-vowel marks from every Kashmiri
target it is given:

| Mark | Codepoint | Removed |
|---|---|---:|
| kasra | U+0650 | 100% |
| damma | U+064F | 100% |
| fatha | U+064E | 100% |

Indic NLP inherits this behaviour from Urdu, where the marks are optional.
Kashmiri is not Urdu. The marks are phonemic, and the competition's own
`kashmirinormalizer` package carries the warning in its source: *"According to
linguists diacritics are important in kashmiri unlike urdu, so don't remove
them."*

26.4% of Kashmiri words carry at least one of the three. A translator that is
otherwise **perfect** but omits only these marks scores:

| Metric | Perfect | Perfect minus the three marks |
|---|---:|---:|
| BLEU | 100 | `56.92` |
| chrF++ | 100 | `79.07` |
| Composite | 100 | **`67.08`** |

So any model trained through that preprocessing carries a hard `0.67x` ceiling,
and no amount of extra data can break it. Our first nine training runs all sat
under it and all scored between `9.42` and `9.69`, whatever we fed them.

The fix is to normalize training targets with `KashmiriNormalizer`, the scorer's
own normalizer, which keeps every Kashmiri mark. Inference never touched
`IndicProcessor` on the target side, so the marks survive to the output file
once the model has learned to write them.

This single change took the score from `9.69` to `13.75`.

### Training data

The fix only helps if the training targets actually carry the marks. Our
existing pool did not: 53.2% of its targets had none of the three, and the
largest sub-pool was 73.2% unmarked. Half the examples would have taught the
model to omit exactly what we were trying to preserve.

So the pool was rebuilt from the two consistently marked sources:

| Property | Value |
|---|---|
| Rows | 111,053 |
| Sources | BPCC (`bpcc_h`), OPUS wikimedia |
| Targets with no marks | 25.0% |
| Validation references, for comparison | 21.8% |

Selecting for mark consistency rather than volume also made each epoch 1,735
steps instead of 4,188.

### Training

Initialized from our own earlier 1B fine-tune, which already translated
Kashmiri competently and only had to learn to write the vowels.

| Setting | Value |
|---|---|
| Optimizer | Adafactor |
| Learning rate | `3e-5`, 200 warmup steps |
| Epochs | 4, so 6,940 steps |
| Batch | 8, gradient accumulation 8, effective 64 |
| Label smoothing | `0.1` |
| Precision | fp16 |
| Target normalizer | `KashmiriNormalizer` |

Adafactor rather than AdamW: AdamW's optimizer state for 1.1B parameters needs
roughly 17.6 GB and does not fit the 15 GB GPU this was trained on.

### Weight averaging

The released weights are the element-wise **mean of two checkpoints**, steps
6,500 and 6,940 of the same run. Averaging is standard in Transformer machine
translation and it beat both of its own components, on validation and on the
leaderboard, which is the expected signature of a real averaging gain rather
than noise. It is worth `+0.15`.

More checkpoints is worse: a four-way average scored `13.66`.

### Decoding

| Setting | Value |
|---|---|
| Beams | 5 |
| `max_length` | 48 |
| Length penalty | `1.0` |
| Return sequences | 1 |

Every one of these was swept on 1,500 held-out rows and these values won. Beam
widths 5, 8 and 12 scored `32.84`, `32.76` and `32.84`. Length penalty `1.2`
scored `32.65` against `32.78`. No decode-time diacritic bias is applied,
because the trained model already emits marks at 102% of reference density and
pushing it higher only adds wrong ones.

Output passes through one guard: a repeated punctuation mark collapses to a
single mark. Degenerate decoding can emit such a run and no reference contains
one. It changed 0 of the 1,730 scored rows.

## Where the score came from

| Change | Gain |
|---|---:|
| Kashmiri diacritics preserved in training targets | **`+4.06`** |
| Two-checkpoint weight averaging | `+0.15` |

`9.69` → `11.99` (partly trained) → `13.75` (fully trained) → `13.90`
(averaged).

## What did not work

Recorded because it is the more useful half of the result. Twelve ideas were
tested and ten failed. Seven were killed on held-out data without spending a
leaderboard submission.

| Idea | Measurement |
|---|---|
| Diacritic bias on the decoder logits | swept `-3` to `+1`; `0` wins |
| Length penalty tuning | `1.2` gives `32.65` against `32.78` |
| Beam width tuning | 5, 8, 12 give `32.84`, `32.76`, `32.84` |
| Minimum Bayes risk within one model | 3 candidates `32.17`, 5 give `31.83` |
| Post-hoc diacritic correction | only 0.9% of words are eligible |
| Corpus-frequency diacritic filling | worse at every threshold |
| Context-aware mark selection | `+0.16`, below a plain unigram's `+0.19` |
| Length-matched domain adaptation pass | every checkpoint scored below the base |
| Medoid consensus across four systems | `13.89` against `13.90` |
| Word-level mark voting across systems | `+0.19` held out, `13.76` on the leaderboard |
| Four-way checkpoint average | `13.66` against `13.90` |

The last two are the instructive ones. Word-level mark voting had an understood
mechanism, a measured oracle worth `+0.62`, stability across sixteen parameter
settings, and a verified guarantee that it never changed a word lexically. It
still lost. Held-out differences below about `0.3` on this task did not survive
to the leaderboard, so a good mechanism is not a substitute for the power to
measure it.

## What remains

Diacritic **accuracy**, not density, is the largest measured gap. Of the words
that match the reference once marks are stripped, 85.4% carry exactly the right
marks. Copying the reference marks onto those words raises the composite from
`33.73` to `40.76`, a factor of about `1.21`, which on `13.90` points at roughly
`16.8`. No decode setting and no context-free lexicon reaches it; both were
tested to exhaustion. It needs mark-aware training or a contextual diacritizer.

## Files

| File | Purpose |
|---|---|
| `load_model.py` | loads the model and tokenizer, and runs generation |
| `infer_single.py` | translates one sentence given on the command line |
| `infer_batch.py` | translates a CSV and writes the submission format |
| `requirements.txt` | pinned dependencies |
| `examples/sample_input.csv` | five sentences in the input format |

## Notes

This repository holds the inference path and the write-up. It does not include
competition data, which the rules do not allow us to redistribute.

The model is a derivative of `ai4bharat/indictrans2-en-indic-1B`; see that model
card for its licence and intended use. Training used BPCC and OPUS wikimedia
English–Kashmiri pairs.
