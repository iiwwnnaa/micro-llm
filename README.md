# Micro-LLM: A minimal language model built from scratch in NumPy

A minimal character-level language model written from scratch using only NumPy. Built to understand the underlying math and mechanics of autoregressive language models without relying on high-level frameworks like PyTorch or TensorFlow.

## Key Features
- **Zero Frameworks:** Pure NumPy implementation for forward pass, loss calculation, and backprop.
- **Stable Softmax:** Applied max-subtraction trick to prevent numerical overflow during exponentiation.
- **Analytic Gradients:** Used the exact derivative of Cross-Entropy Loss w.r.t. Softmax logits (`probs - target`) for weight updates.

## Roadmap

- [x] **01_single_char_model**
  - Predicting the next character from a single input character ($4 \times 4$ weight matrix)
  - Gradient descent implementation using pure NumPy
- [ ] **02_multi_char_context** *(Next)*
  - Expanding context window to $N$ previous characters
  - Building character embedding lookups and matrix concatenation from scratch
- [ ] **03_word_level_tokenizer**
  - Transitioning from character-level to word/token-level vocabulary and building a basic tokenizer
- [ ] **04_sequence_models_from_scratch**
  - Implementing RNN / LSTM cells in NumPy to handle variable-length context sequences
  - *(Optional)* Building a minimal Self-Attention block to understand Transformer fundamentals
- [ ] **05_pytorch_refactoring**
  - Porting the NumPy implementation to PyTorch for scalable training and GPU support
