## **DeepSeek v3 Components in PyTorch**

🚀 **Implementing fundamental components from DeepSeek v3 research paper in PyTorch.**  
This project includes **Sparse Mixture of Experts (MoE) with Auxiliary-Loss-Free Load Balancing**, **Rotary Positional Embeddings (RoPE)**, **Multi-Token Prediction Modules (MTP Modules)** and **Multi-Head Latent Attention (MLA)** implemented from scratch.

---

## 📂 **Project Structure**

| File           | Description                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------|
| **`model.py`**  | Integrates each and every component together to form the main model.                                |
| **`MLAwRoPE.py`** | Implements Multi-Head Latent Attention integrated with Rotary Positional Embeddings.         |
| **`DS_MoE.py`**   | Implements Sparse Mixture of Experts (MoE) with Auxiliary-Loss-Free Load balancing and fundamental features like gating and top-k routing.|
| **`TransformerBlock.py`**   | Implements Transformer Block by combining MLAwRoPE.py and DS_MoE.py along with RMSNorm and Residual connenctions. |
| **`MTP_module.py`**   | Implements Multi Token Prediction using several MTP modules. |

---

## 🌟 **Future Work**
- **Auxiliary Loss-Free Load Balancing:** Optimize MoE without auxiliary loss terms.                                                         *(implementation finished)*
- **Multitoken Prediction(MTP):** Optimize the model to predict future tokens, essentially helping the main model and increasing efficiency. *(implementation finished)*

---

## 🧪 **Features Implemented**
- **Mixture of Experts (MoE)**: A scalable and modular design for efficiently handling large models with groups of Experts.
- **Auxiliary-Loss-Free Load Balancing**: Ensures balance in each expert's load.
- **Rotary Positional Embeddings (RoPE)**: Enhancing attention mechanisms with rotational encodings.
- **Multi-Head Latent Attention (MLA)**: Aggregating multiple attention heads with latent spaces for efficient feature extraction.
- **Multi-Token Prediction(MTP)**: Aggregating multiple MTP modules with increasing depths to predict future tokens outside the scope of the main model and increasing overall prediction efficieny of the model.
---

## 📜 **References**
1. [DeepSeek v3 Research Paper](#) _(https://arxiv.org/pdf/2412.19437)._

---

## 🔥 **Get Involved**
Feel free to fork the repo, raise issues, or submit pull requests for contributions!

---

