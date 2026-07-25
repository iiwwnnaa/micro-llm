import numpy as np

# Data setup
text = "hello"
chars = sorted(list(set(text)))  # ['e', 'h', 'l', 'o']
vocab_size = len(chars)

# Char to index mapping
char_to_ix = {ch: i for i, ch in enumerate(chars)}

# Training pair: input 'h' -> target 'e'
input_idx = char_to_ix['h']   # index 1
target_idx = char_to_ix['e']  # index 0

np.random.seed(42)

# Weight matrix W (4x4)
# Row: input char / Col: score (logit) for next predicted char
W = np.random.randn(vocab_size, vocab_size)

def softmax(x):
    # Subtract max to prevent exp overflow
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

learning_rate = 0.5

# Training loop
for epoch in range(100):
    logits = W[input_idx]
    probs = softmax(logits)

    # Gradient of Cross-Entropy Loss w.r.t. Softmax logits (probs - target)
    d_scores = probs.copy()
    d_scores[target_idx] -= 1.0  # Push target score up, non-targets down

    # Update weights for the current input char row
    W[input_idx] -= learning_rate * d_scores

# Test inference
final_logits = W[input_idx]
final_probs = softmax(final_logits)

print(f"Target ('e') Probability: {final_probs[target_idx] * 100:.2f}%")
print(f"Predicted Character: '{chars[np.argmax(final_probs)]}'")