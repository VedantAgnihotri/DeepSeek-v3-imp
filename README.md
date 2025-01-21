DeepSeek v3 Components in PyTorch
This repository contains an implementation of DeepSeek v3 fundamental components in PyTorch, based on its research paper. The project focuses on implementing core modules, such as Mixture of Experts (MoE), Rotary Positional Embeddings (RoPE), and Multi-Head Latent Attention (MLA), from scratch.

Project Structure
The repository is organized into the following files:

main.py

The entry point for testing the implemented components.
Demonstrates how the Mixture of Experts and Rotary Positional Embeddings integrate with the Multi-Head Latent Attention module.
Includes input preparation and forward pass execution.
MLAwRoPE.py

Implements Multi-Head Latent Attention (MLA) combined with Rotary Positional Embeddings (RoPE).
Features:
Positional encoding using RoPE for capturing sequence relationships.
Scaled dot-product attention using latent query and input key-value pairs.
Supports flexible attention heads and embedding dimensions.
DS_MoE.py

Implements the DeepSeek Mixture of Experts (DS_MoE) module.
Features:
Routed and shared experts for specialized and shared computations.
Routing mechanism based on gating and top-k expert selection.
Key Features
Mixture of Experts (MoE):
Efficiently routes input through selected experts, optimizing computational resources while maintaining performance. Both routed and shared experts are supported.

Rotary Positional Embeddings (RoPE):
Enhances positional encoding by embedding spatial relationships directly into the input using sinusoidal functions.

Multi-Head Latent Attention (MLA):
Leverages latent queries to process complex input sequences and integrates attention mechanisms with RoPE.

Future Scope
Auxiliary Loss-Free Load Balancing:
Plan to implement a mechanism to balance expert loads without auxiliary loss terms.

Multitoken Prediction (MTP):
Extend the current modules to include multitask learning objectives, enabling the model to generalize across tasks.

Scaling and Optimization:
Optimize the current architecture for larger datasets, distributed training, and real-world applications.

References
https://arxiv.org/pdf/2412.19437
PyTorch Documentation

Stay tuned for updates as I explore additional components and techniques from the DeepSeek v3 architecture! 🎉
