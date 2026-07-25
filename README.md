# AHL-SyncMark with FAMA-D: DNN Ownership Verification

This project implements **AHL-SyncMark**, a hybrid deep neural network (DNN) ownership-verification protocol described in `AHL-SyncMark_IEEE_12pg.docx`. It features zero clean classification accuracy (CACC) drop, a multi-bit signature, and privacy-preserving blockchain auditability.

In addition, it implements a novel, unique algorithm extension called **FAMA-D** (Fisher-Attribution Manifold Alignment for Distillation) designed to address the principal weaknesses identified in the paper: robustness under knowledge distillation and security against adaptive, FIM-aware adversaries.

## Key Features

1. **Stable Subspace Selection**: Intermediate channels are ranked using a diagonal Fisher Information Matrix (FIM) curvature estimate.
2. **Auxiliary Hidden Layer (AHL)**: A private signature decoder maps stable activations to a $d$-bit signature under a strict stop-gradient constraint (CACC drop is 0.00%).
3. **Explanation-as-a-Watermark (EaaW)**: The host model's gradient attribution maps are shaped on a trigger input using a combined MSE and Pearson correlation loss toward a target mask.
4. **zk-SNARK Sim Registry**: Simulates blockchain commitments and zero-knowledge proof generation/verification to audit ownership without exposing the signature, decoder, or weights.
5. **FAMA-D Algorithm (Novel Extension)**:
   - **Cross-Layer Activation Distillation Anchor (CLADA)**: Projects the signature onto soft logits using a regularizer during teacher training, forcing any student model trained via knowledge distillation to replicate the signature in its output distribution.
   - **Chaotic Space Keying (CSK)**: Scrambles the selection of stable channels using an orthonormal projection matrix derived from a chaotic Henon map seeded by a private key, preventing FIM-aware adversaries from locating or pruning watermarked activations.

## File Structure

- `dataset.py`: Procedural generator for the synthetic 10-class dataset (features: shape, color, texture).
- `model.py`: ResNet-style classifier utilizing residual blocks with Layer Normalization and hooks.
- `watermark_core.py`: Implementation of diagonal FIM calculation, gradient-isolated AHL decoder, and EaaW (SmoothGrad Integrated Gradients).
- `fama_d.py`: The FAMA-D algorithm implementations (CLADA mapping, CSK projection generator, and student distillation).
- `zk_registry.py`: zk-SNARK registry simulation.
- `main.py`: Full execution script running baseline training, watermarking, attack simulations (pruning, quantization, fine-tuning, knowledge distillation), and verification.

## Running the Code

Execute the pipeline using python:
```bash
python main.py
```
