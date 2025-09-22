# Hybrid Decoding for Canary v2 ASR Models

This repository provides an unofficial implementation of **Hybrid Decoding** for **NVIDIA Canary v2** models, combining:  
- AED (Attention Encoder-Decoder) models, and  
- CTC (Connectionist Temporal Classification) models.  

---

## 📌 Background

- [Canary v2](https://huggingface.co/nvidia/canary-1b-v2) public models support **25 languages** with both AED and CTC backends.  
- This work applies the **[hybrid decoding approach](https://arxiv.org/abs/2508.19671)**, but unlike the original paper, it is based on a design **without a shared encoder**.  
- Model outputs include Punctuation & Capitalization (PnC). Due to PnC, frequent mismatches occur between the CTC and Transformer decoder outputs, so the patch length is fixed to 1.
- In the current implementation, **larger batch sizes do not guarantee an improvement in inference speed**, as the number of forward pass is determined by the worst-case sample in the batch. (Recommended: 1, 2, 4)
- The released logic currently supports **greedy search only**.  
- The difference in word error rate between Canary v2 and hybrid decoding is **less than 0.01%**.
---

## ⚡ Benchmark Results on NVIDIA V100

### 🗂 FLEURS en-US Testset
- **Dataset:** [FLEURS en-US](https://huggingface.co/datasets/google/fleurs)  
- **Samples:** 647  
- **Total Audio Duration:** 106.47 min

| Mode        | Inference Method                  | Time (s) | Speedup |
|-------------|-----------------------------------|----------|---------|
| **Single**  | Canary v2                         | 205.83   | x1.00   |
| **Single**  | Canary v2 + CTC Forced Alignment  | 264.64   | x0.78   |
| **Single**  | Canary v2 Hybrid                  | 85.24    | x2.41   |
| **Single**  | Canary v2 Hybrid + CTC Forced Alignment | 114.37 | x1.80   |
| **Batch (16)** | Canary v2                      | 75.47    | x2.73   |
| **Batch (16)** | Canary v2 + CTC Forced Alignment | 111.76  | x1.84   |
| **Batch (4)**  | Canary v2 Hybrid               | 63.04    | x3.27   |
| **Batch (4)**  | Canary v2 Hybrid + CTC Forced Alignment | 75.56 | x2.72   |


---

### 🎬 Long-form Audio (Chunking Mechanism)
processes a single long-form audio input, splits it into chunks.

#### [TED English](https://www.youtube.com/watch?v=y9Trdafp83U) (12m 18s)

| Inference Method                        | Time (s) | Speedup |
|-----------------------------------------|----------|----------------------------------|
| **Canary v2**                           | 9.47     | x1.00                            |
| **Canary v2 + CTC Forced Alignment**    | 11.95    | x0.79                            |
| **Canary v2 Hybrid**                    | 7.66     | x1.24                            |
| **Canary v2 Hybrid + CTC Forced Alignment** | 8.11    | x1.17                            |

#### [TED French](https://www.youtube.com/watch?v=0u7tTptBo9I) (13m 19s)

| Inference Method                        | Time (s) | Speedup |
|-----------------------------------------|----------|----------------------------------|
| **Canary v2**                           | 12.38    | x1.00                            |
| **Canary v2 + CTC Forced Alignment**    | 15.21    | x0.81                            |
| **Canary v2 Hybrid**                    | 8.55     | x1.45                            |
| **Canary v2 Hybrid + CTC Forced Alignment** | 8.98    | x1.38                            |



**Canary v2:**, performs batch inference on those chunks.

**Canary v2 Hybrid:** performs batch inference for the entire CTC model and the encoder of the AED model, while the transformer decoder is run with single inference. (AUTO)

---

## 🚀 Usage Example

```bash
wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
```

```python
from nemo.collections.asr.models import ASRModel

model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")

def inference(audio, lang, timestamps=False):

    # Transcribe
    output = model.transcribe(
        audio, 
        source_lang=lang, 
        target_lang=lang, 
        timestamps=timestamps, 
        batch_size=4
    )
    
    # Print results
    if timestamps:
        word_timestamps = output[0].timestamp['word']
        segment_timestamps = output[0].timestamp['segment']

        for stamp in segment_timestamps:
            print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")
    else:
        print(output[0].text)

# Example usage
audio = ['2086-149220-0033.wav']
inference(audio, "en", True)
```

---

## 🔧 Installation

```bash
# Base image
docker pull nvcr.io/nvidia/pytorch:25.01-py3

# Dependencies
apt update && apt install ffmpeg

# NeMo Toolkit
pip install -U nemo_toolkit['asr']

# clone this repository
git clone https://github.com/turieo/faster-canary.git
cd faster-canary
```

---

## Citation
```bibtex
@article{lim2025hybrid,
  title={Hybrid Decoding: Rapid Pass and Selective Detailed Correction for Sequence Models},
  author={Yunkyu Lim and Jihwan Park and Hyung Yong Kim and Hanbin Lee and Byeong-Yeol Kim},
  journal={arXiv preprint arXiv:2508.19671},
  year={2025}
}

@article{seko2025canary,
  title={Canary-1B-v2 \& Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST},
  author={Monica Sekoyan and Nithin Rao Koluguri and Nune Tadevosyan and Piotr Zelasko and Travis Bartley and Nick Karpov and Jagadeesh Balam and Boris Ginsburg},
  journal={arXiv preprint arXiv:2509.14128},
  year={2025}
}
```

