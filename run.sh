#!/bin/bash

export WANDB_MODE=offline
mkdir -p logs

echo "[1/18] Starting Baseline..."
python train_model.py --config configs/baseline.yaml > logs/baseline.log 2>&1
echo "[1/18] Baseline completed!"

echo "[2/18] Starting Baseline + Augmentation..."
python train_model.py --config configs/baseline_aug.yaml > logs/baseline_aug.log 2>&1
echo "[2/18] Baseline + Augmentation completed!"

echo "[3/18] Starting Softmax Attention..."
python train_model.py --config configs/softmax_attention.yaml > logs/softmax_attention.log 2>&1
echo "[3/18] Softmax Attention completed!"

echo "[4/18] Starting Softmax Attention + Augmentation..."
python train_model.py --config configs/softmax_attention_aug.yaml > logs/softmax_attention_aug.log 2>&1
echo "[4/18] Softmax Attention + Augmentation completed!"

echo "[5/18] Starting Softmax Attention Fine-tune..."
python train_model.py --config configs/softmax_attention_finetune.yaml > logs/softmax_attention_finetune.log 2>&1
echo "[5/18] Softmax Attention Fine-tune completed!"

echo "[6/18] Starting Softmax Attention Fine-tune + Augmentation..."
python train_model.py --config configs/softmax_attention_finetune_aug.yaml > logs/softmax_attention_finetune_aug.log 2>&1
echo "[6/18] Softmax Attention Fine-tune + Augmentation completed!"

echo "[7/18] Starting SE Attention..."
python train_model.py --config configs/se_attention.yaml > logs/se_attention.log 2>&1
echo "[7/18] SE Attention completed!"

echo "[8/18] Starting SE Attention + Augmentation..."
python train_model.py --config configs/se_attention_aug.yaml > logs/se_attention_aug.log 2>&1
echo "[8/18] SE Attention + Augmentation completed!"

echo "[9/18] Starting SE Attention Fine-tune..."
python train_model.py --config configs/se_attention_finetune.yaml > logs/se_attention_finetune.log 2>&1
echo "[9/18] SE Attention Fine-tune completed!"

echo "[10/18] Starting SE Attention Fine-tune + Augmentation..."
python train_model.py --config configs/se_attention_finetune_aug.yaml > logs/se_attention_finetune_aug.log 2>&1
echo "[10/18] SE Attention Fine-tune + Augmentation completed!"

echo "[11/18] Starting CBAM..."
python train_model.py --config configs/cbam.yaml > logs/cbam.log 2>&1
echo "[11/18] CBAM completed!"

echo "[12/18] Starting CBAM + Augmentation..."
python train_model.py --config configs/cbam_aug.yaml > logs/cbam_aug.log 2>&1
echo "[12/18] CBAM + Augmentation completed!"

echo "[13/18] Starting CBAM Fine-tune..."
python train_model.py --config configs/cbam_finetune.yaml > logs/cbam_finetune.log 2>&1
echo "[13/18] CBAM Fine-tune completed!"

echo "[14/18] Starting CBAM Fine-tune + Augmentation..."
python train_model.py --config configs/cbam_finetune_aug.yaml > logs/cbam_finetune_aug.log 2>&1
echo "[14/18] CBAM Fine-tune + Augmentation completed!"

echo "[15/18] Starting Self-Attention..."
python train_model.py --config configs/self_attention.yaml > logs/self_attention.log 2>&1
echo "[15/18] Self-Attention completed!"

echo "[16/18] Starting Self-Attention + Augmentation..."
python train_model.py --config configs/self_attention_aug.yaml > logs/self_attention_aug.log 2>&1
echo "[16/18] Self-Attention + Augmentation completed!"

echo "[17/18] Starting Self-Attention Fine-tune..."
python train_model.py --config configs/self_attention_finetune.yaml > logs/self_attention_finetune.log 2>&1
echo "[17/18] Self-Attention Fine-tune completed!"

echo "[18/18] Starting Self-Attention Fine-tune + Augmentation..."
python train_model.py --config configs/self_attention_finetune_aug.yaml > logs/self_attention_finetune_aug.log 2>&1
echo "[18/18] Self-Attention Fine-tune + Augmentation completed!"

echo "All 18 experiments completed!"