# Research Report: Adapting the Hierarchical Reasoning Model (HRM) for Chest X-Ray Analysis

## 1. Introduction

The Hierarchical Reasoning Model (HRM), developed by Sapient Inc., is a novel recurrent architecture designed for complex, sequential reasoning tasks. Inspired by the hierarchical processing of the human brain, HRM combines a high-level module for abstract planning with a low-level module for rapid, detailed computation. Its key strengths—a compact size (27M parameters), exceptional performance on reasoning benchmarks like ARC-AGI with minimal training data, and its ability to perform complex tasks in a single forward pass—make it a compelling candidate for adaptation to other domains.

This report outlines a strategy for adapting the HRM architecture to a challenging medical imaging task: the classification of 18 different pathologies from 224x224 grayscale chest x-ray images.

## 2. Proposed Model Adaptation for X-Ray Data

The existing HRM checkpoint for ARC is trained on 30x30 grids with 10 unique values (colors). Adapting this to process standard chest x-ray images requires specific, but minimal, changes to the model's input configuration. The core reasoning logic of the model remains unchanged.

### Key Configuration Changes:

The model's structure is defined by the `HierarchicalReasoningModel_ACTV1Config`. The following parameters must be modified:

1.  **`vocab_size` (Vocabulary Size):**
    *   **Current:** `10` (for the 10 colors in the ARC dataset).
    *   **Proposed:** `256` (to represent the full range of an 8-bit grayscale pixel, 0-255).
    *   **Impact:** This change requires resizing the model's input token embedding layer (`embed_tokens`) and its output projection layer (`lm_head`).

2.  **`seq_len` (Sequence Length):**
    *   **Current:** `900` (from a flattened 30x30 grid).
    *   **Proposed:** `50176` (from a flattened 224x224 image).
    *   **Impact:** The model utilizes Rotary Position Embeddings (RoPE), which can dynamically adjust to different sequence lengths without requiring changes to model weights. This makes HRM highly flexible for various input sizes.

## 3. Analysis of Model Growth

A primary concern when adapting models for high-resolution images is the potential for a massive increase in parameter count. **HRM's architecture elegantly avoids this issue.**

The core reasoning modules (`H_level` and `L_level`), which contain the bulk of the model's 27 million parameters, are independent of the input sequence length and vocabulary size. The only components that scale are the input and output layers.

*   **Calculation of Parameter Increase:**
    *   The `hidden_size` of the model is 512.
    *   **Token Embedding Layer (`embed_tokens`):**
        *   Old size: `10 * 512 = 5,120` parameters.
        *   New size: `256 * 512 = 131,072` parameters.
    *   **Output Layer (`lm_head`):**
        *   Old size: `512 * 10 = 5,120` parameters.
        *   New size: `512 * 256 = 131,072` parameters.
    *   **Total Increase:** The total number of new parameters is approximately `(131,072 - 5,120) * 2 ≈ 252,000`.

A 252k parameter increase on a 27M parameter model represents a growth of **less than 1%**. This demonstrates that HRM can be adapted to handle significantly larger and more complex input data without a substantial increase in model size.

## 4. Proposed Training Strategy

### 4.1. Sequential Patching for Computational Efficiency

While HRM *can* handle a flattened 50,176-element sequence, the quadratic complexity of the self-attention mechanism (`O(n^2)`) would make this computationally expensive. A more effective approach is to leverage the model's inherent recurrent nature.

**Proposed Method:**
1.  **Divide the Image:** Split the 224x224 x-ray into two vertical **224x112 patches**.
2.  **Flatten and Feed Sequentially:**
    *   **Step 1:** Feed the first flattened patch (a sequence of `25,088` pixels) into the model.
    *   **Step 2:** Capture the model's internal recurrent state (the `carry` object, containing `z_H` and `z_L` tensors).
    *   **Step 3:** Feed the second flattened patch into the model, but initialize it with the `carry` from the previous step.
3.  **Final Prediction:** The model's final output after processing the last patch would represent its reasoning over the entire image.

This method keeps the `seq_len` manageable, significantly reducing memory and compute requirements while allowing the model to build a holistic understanding of the full image.

### 4.2. Dataset Formatting and `puzzle_emb`

For a training set of 1,000 x-rays across 18 pathologies:

*   **Input Data:** Each x-ray would be a sequence (or a sequence of patches) of pixel values from 0-255.
*   **Labels:** The target output could be a classification token representing one of the 18 pathologies.

**Leveraging `puzzle_emb`:**
The `puzzle_emb` is a powerful feature that learns a unique vector for each "puzzle" or task instance. In this medical context, it could be adapted in several ways:
*   **Pathology-specific Embeddings:** Assign a unique, learnable `puzzle_identifier` to each of the 18 pathologies. This would provide the model with a strong, explicit signal about the classification target, potentially allowing it to learn distinct features for each condition more effectively.
*   **Patient-level Embeddings:** If multiple images exist per patient, a unique embedding could be assigned to each patient to help the model track progression or patient-specific anatomical features.

## 5. Conclusion

The Hierarchical Reasoning Model (HRM) is a highly promising architecture for complex medical imaging tasks like chest x-ray analysis. Its key advantages are:
*   **Scalability:** It can be adapted to high-resolution images with a negligible (<1%) increase in model size.
*   **Efficiency:** Its recurrent design supports sequential patching, enabling the processing of large images with manageable computational cost.
*   **Data-Efficiency:** Its success on ARC with only 1,000 training samples suggests it could be effectively trained on similarly sized medical datasets.

By modifying the input configuration and leveraging its inherent architectural strengths, HRM presents a feasible and innovative path toward building powerful, efficient, and interpretable models for medical diagnostics.
