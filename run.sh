#!/bin/bash

mkdir -p logs

echo "[1/6] Starting Baseline..."
python train_model.py --experiment baseline --epochs 30 > logs/baseline.log 2>&1
echo "[1/6] Baseline completed!"

echo "[2/6] Starting Baseline + Augmentation..."
python train_model.py --experiment baseline_aug --epochs 30 > logs/baseline_aug.log 2>&1
echo "[2/6] Baseline + Augmentation completed!"

echo "[3/6] Starting Softmax Attention..."
python train_model.py --experiment softmax_attention --epochs 30 > logs/softmax_attention.log 2>&1
echo "[3/6] Softmax Attention completed!"

echo "[4/6] Starting Softmax Attention Fine-tune..."
python train_model.py --experiment softmax_attention_finetune --epochs 30 > logs/softmax_attention_finetune.log 2>&1
echo "[4/6] Softmax Attention Fine-tune completed!"

echo "[5/6] Starting SE Attention..."
python train_model.py --experiment se_attention --epochs 30 > logs/se_attention.log 2>&1
echo "[5/6] SE Attention completed!"

echo "[6/6] Starting SE Attention + Augmentation..."
python train_model.py --experiment se_attention_aug --epochs 30 > logs/se_attention_aug.log 2>&1
echo "[6/6] SE Attention + Augmentation completed!"

echo "All experiments completed!"