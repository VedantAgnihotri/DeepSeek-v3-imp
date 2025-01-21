## **DeepSeek v3 Components in PyTorch**

🚀 **Implementing fundamental components from DeepSeek v3 research paper in PyTorch.**  
This project includes **Mixture of Experts (MoE)**, **Rotary Positional Embeddings (RoPE)**, and **Multi-Head Latent Attention (MLA)** implemented from scratch.

---

## 📂 **Project Structure**

| File           | Description                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------|
| **`main.py`**  | Entry point for testing components and running forward passes.                                |
| **`MLAwRoPE.py`** | Implements Multi-Head Latent Attention integrated with Rotary Positional Embeddings.         |
| **`DS_MoE.py`**   | Implements Mixture of Experts (MoE) with fundamental features like gating and top-k routing.|

---

## 🌟 **Future Work**
- **Auxiliary Loss-Free Load Balancing:** Optimize MoE without auxiliary loss terms.  
- **Multitoken Prediction(MTP):** Optimize the model to predict future tokens, essentially helping the main model and increasing efficiency.  

---

## 🧪 **Features Implemented**
- **Mixture of Experts (MoE)**: A scalable and modular design for efficiently handling large models.
- **Rotary Positional Embeddings (RoPE)**: Enhancing attention mechanisms with rotational encodings.
- **Multi-Head Latent Attention (MLA)**: Aggregating multiple attention heads for better feature extraction.

---

## 📜 **References**
1. [DeepSeek v3 Research Paper](#) _(https://arxiv.org/pdf/2412.19437)._

---

## 🔥 **Get Involved**
Feel free to fork the repo, raise issues, or submit pull requests for contributions!

---

