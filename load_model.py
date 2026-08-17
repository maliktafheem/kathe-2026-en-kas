"""Load the KATHE 2026 English to Kashmiri translation model.

The inference scripts import `load()` from here. You can also run this file
directly to check that the weights load and that one sentence translates:

    python load_model.py
"""

from __future__ import annotations

MODEL_ID = "maliktafheem/kathe-2026-en-kas-arab-1b"
SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"

# Decoding settings used for the scored submission. Each one was swept on a
# 1,500-row validation set; these values won. See the README.
NUM_BEAMS = 5
MAX_LENGTH = 48
LENGTH_PENALTY = 1.0
BATCH_SIZE = 16


class Translator:
    """A loaded model, its tokenizer, and the IndicTrans2 text processor."""

    def __init__(self, model, tokenizer, processor, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device

    def translate(
        self,
        sentences: list[str],
        *,
        batch_size: int = BATCH_SIZE,
        max_length: int = MAX_LENGTH,
        num_beams: int = NUM_BEAMS,
        length_penalty: float = LENGTH_PENALTY,
        progress: bool = False,
    ) -> list[str]:
        """Translate English sentences into Kashmiri, in the given order."""
        import torch

        translations: list[str] = []
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            processed = self.processor.preprocess_batch(
                batch, src_lang=SRC_LANG, tgt_lang=TGT_LANG
            )
            inputs = self.tokenizer(
                processed,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=max_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    num_return_sequences=1,
                )
            # IndicTrans2 has separate source and target vocabularies, so the
            # output ids must be decoded in target space.
            with self.tokenizer.as_target_tokenizer():
                decoded = self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            translations.extend(self.processor.postprocess_batch(decoded, lang=TGT_LANG))
            if progress:
                print(
                    f"  translated {len(translations)} / {len(sentences)}",
                    flush=True,
                )
        return translations


def load(model_id: str = MODEL_ID, device: str | None = None) -> Translator:
    """Load the model and return a `Translator`.

    `device` defaults to CUDA when a GPU is visible, otherwise CPU. On CPU the
    model runs in float32 and needs about 5 GB of RAM.
    """
    import torch
    from IndicTransToolkit import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    )
    # Generation reads `config.vocab_size`. For this architecture that must be
    # the target vocabulary, which is a different size from the source one.
    model.config.vocab_size = model.config.decoder_vocab_size

    # A weight left on the meta device loads without error and then translates
    # to noise, so fail here instead.
    unloaded = [name for name, p in model.named_parameters() if p.is_meta]
    if unloaded:
        raise RuntimeError(f"These weights did not load: {unloaded[:5]}")

    model.to(device)
    if device == "cuda":
        # The scored submission was generated in float16 on a GPU.
        model.half()
    model.eval()
    return Translator(model, tokenizer, IndicProcessor(inference=True), device)


if __name__ == "__main__":
    import sys

    # A Windows console defaults to a codepage that cannot represent Arabic
    # script, so printing a translation would raise UnicodeEncodeError.
    sys.stdout.reconfigure(encoding="utf-8")

    translator = load()
    print(f"loaded {MODEL_ID} on {translator.device}")
    for sentence in ["The sun rises in the east.", "My brother is a teacher."]:
        print(f"{sentence}  ->  {translator.translate([sentence])[0]}")
