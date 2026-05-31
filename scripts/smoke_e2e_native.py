#!/usr/bin/env python3
"""
End-to-end smoke test for the native (torchtext-free) loader + training loop.

It generates a tiny synthetic dataset in a temp dir, writes a minimal config,
and runs training for 1 epoch with translation-only (no TF dependency).
"""

from __future__ import annotations

import gzip
import os
import pickle
import tempfile
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signjoey.training import train


def _write_split(root: str, name: str, samples):
    path = os.path.join(root, name)
    with gzip.open(path, "wb") as f:
        pickle.dump(samples, f)
    return path


def main():
    tmpdir = tempfile.mkdtemp(prefix="slt-e2e-native-")

    feature_size = 32
    time_steps = 40
    samples = []
    for i in range(8):
        samples.append(
            {
                "name": f"seq{i}",
                "signer": "s0",
                "gloss": "A B C",
                "text": "hello world",
                # expected shape [F, T]
                "sign": torch.randn(feature_size, time_steps),
            }
        )

    train_file = _write_split(tmpdir, "toy.pami0.train", samples)
    dev_file = _write_split(tmpdir, "toy.pami0.dev", samples[:4])
    test_file = _write_split(tmpdir, "toy.pami0.test", samples[:4])

    cfg = f"""
name: toy_native_smoke
data:
  loader: native
  data_path: {tmpdir}
  version: toy
  train: {os.path.basename(train_file)}
  dev: {os.path.basename(dev_file)}
  test: {os.path.basename(test_file)}
  feature_size: {feature_size}
  level: word
  txt_lowercase: true
  max_sent_length: 400
  random_train_subset: -1
  random_dev_subset: -1
  gls_voc_limit: 50
  gls_voc_min_freq: 1
  txt_voc_limit: 50
  txt_voc_min_freq: 1
training:
  device: auto
  model_dir: {os.path.join(tmpdir, "model")}
  overwrite: true
  random_seed: 42
  batch_size: 2
  epochs: 1
  validation_freq: 1
  logging_freq: 1
  batch_type: sentence
  optimizer: adam
  learning_rate: 0.0005
  scheduling: plateau
  patience: 2
  decrease_factor: 0.7
  learning_rate_min: 1.0e-7
  weight_decay: 0.0
  label_smoothing: 0.0
  # translation-only to avoid TensorFlow dependency in decoding
  recognition_loss_weight: 0.0
  translation_loss_weight: 1.0
  eval_metric: bleu
  translation_normalization: batch
  translation_max_output_length: 10
  eval_translation_beam_size: 1
  eval_translation_beam_alpha: -1
model:
  initializer: xavier
  bias_initializer: zeros
  init_gain: 1.0
  embed_initializer: xavier
  embed_init_gain: 1.0
  tied_softmax: false
  encoder:
    type: transformer
    num_layers: 1
    num_heads: 2
    embeddings:
      embedding_dim: 64
      dropout: 0.1
      norm_type: batch
    hidden_size: 64
    ff_size: 128
    dropout: 0.1
  decoder:
    type: transformer
    num_layers: 1
    num_heads: 2
    embeddings:
      embedding_dim: 64
      dropout: 0.1
      norm_type: batch
    hidden_size: 64
    ff_size: 128
    dropout: 0.1
"""

    cfg_path = os.path.join(tmpdir, "toy_native.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg)

    # Run training (will also run test() at the end)
    train(cfg_file=cfg_path, device="auto")


if __name__ == "__main__":
    main()
