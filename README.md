# English-to-Urdu Neural Machine Translation with a Transformer

A PyTorch encoder-decoder Transformer trained from scratch on a parallel English-Urdu corpus with SentencePiece subwords, evaluated with sacreBLEU, plus experiments with BPE tokenizers, mBART-50 back-translation and an attempted mBART fine-tune.

**Case study:** [Read it](https://www.laibakhawar.com/work/english-urdu-translation)

## How it works

All code is in `MT_EngtoUrdu.ipynb` (run on Google Colab with a GPU).

**Data (cell 2).** `english-corpus.txt` and `urdu-corpus.txt` are line-aligned (24,525 lines each). They are split 90/10 with `train_test_split(test_size=0.1)`.

**Tokenization (cell 3).** Separate SentencePiece models for English and Urdu, trained on the training split only, `vocab_size=3000` each, with pad/unk/bos/eos ids 0/1/2/3.

**Model (cell 5).** `torch.nn.Transformer` with:

- `d_model=512`, 8 attention heads, 4 encoder layers and 4 decoder layers.
- Separate source and target embeddings scaled by `sqrt(d_model)`, and a linear output layer.
- No positional encoding is added to the embeddings.

**Training (cells 6 and 7).** Adam, learning rate 1e-4, cross-entropy ignoring padding, gradient clipping at norm 0.7, batch size 32, 10 epochs (690 batches per epoch), teacher forcing.

**Decoding and evaluation (cells 8 and 9).** Greedy decoding up to 50 tokens, then sacreBLEU `corpus_bleu` on the first 50 sentences of the held-out split.

**Additional experiments.**

- Cell 12: Hugging Face `tokenizers` BPE models (vocab 3000) trained for both languages. They are saved but not used by the model.
- Cell 14: back-translation with `facebook/mbart-large-50-many-to-many-mmt`: the first 100 English training lines are translated to Urdu and back to English, and the pairs are written, together with the training split, to `final_en.txt`, `final_ur.txt` and `final_data.csv` (22,171 lines).
- Cell 17: fine-tuning mBART-50 on the combined data with `Seq2SeqTrainer`. The run failed with CUDA out of memory on the Colab GPU (the code is now commented out but the error output remains).

## Results

| Metric | Value | Source |
| --- | --- | --- |
| sacreBLEU on 50 held-out sentences | 20.41 | cell 9 |
| Training loss, epoch 1 | 4.16 | cell 7 |
| Training loss, epoch 10 | 0.73 | cell 7 |

The notebook also prints an NLTK BLEU of 0.815 in cell 16. That number is not a valid test score: it is computed on 100 sentences sampled from `final_data.csv`, which is made of the training data plus the back-translated pairs, so the model has seen those sentences in training.

## Repository contents

- `MT_EngtoUrdu.ipynb`: the full pipeline.
- `english-corpus.txt`, `urdu-corpus.txt`: aligned parallel corpus.
- `requirements.txt`: jupyter, nltk, pandas, sacrebleu, scikit-learn, sentencepiece, tokenizers, torch, tqdm, transformers.

## Running it

The notebook was run on Google Colab with a GPU (Python 3.11 according to its outputs). A GPU is strongly recommended; each training epoch took about 35 seconds there.

```bash
git clone https://github.com/LaibaKhawar/English-to-Urdu-Machine-Translation-.git
cd English-to-Urdu-Machine-Translation-
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook MT_EngtoUrdu.ipynb
```

The corpus files are read from the working directory. The notebook writes `train_en.txt`, `train_ur.txt`, the SentencePiece models (`en.model`, `ur.model`), `bpe_en.json`, `bpe_ur.json`, `final_en.txt`, `final_ur.txt` and `final_data.csv` there. The back-translation cell downloads mBART-50 (several GB). `datasets` is used by the commented-out fine-tuning cell but is not in `requirements.txt`.

## Known limitations

- **Causal mask on the encoder.** `forward` passes `generate_square_subsequent_mask` as `src_mask`, so each source token can only attend to earlier source tokens. The encoder should see the whole sentence. Greedy decoding in cell 8 calls the Transformer without that mask, so training and inference also differ.
- **No positional encoding.** The model gets no explicit information about token positions.
- **Embedding size mismatch.** The model is built with vocabularies of 8000 although the SentencePiece models have 3000 pieces; the extra rows are unused.
- **Unseeded split.** `train_test_split` has no `random_state`, so the held-out set and the BLEU score change between runs.
- **Small evaluation.** BLEU is computed on only 50 sentences.
- **Back-translation is barely used.** Only 100 lines were back-translated, and the Transformer was not retrained on the augmented data.
- **mBART fine-tuning did not run** (CUDA out of memory).
- No padding masks are passed to the Transformer, so padded positions are attended to.

## Author

[Laiba Khawar](https://github.com/LaibaKhawar) · [LinkedIn](https://www.linkedin.com/in/laibakhawar/)
