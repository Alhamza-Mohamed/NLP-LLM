# NLP using sequential models

## Neural Networks

![alt text](image.png)

Every word in the training set has a corresponding integer and each tweet has a vector representation of all words inside it.  
The size of the vector representation is = to the size of the longest tweet.  
Every other tweet have 0s to match the size of the longest tweet which known as **padding.**  

---

## Neural Networks layers  

- Dense layer: A layer where each Neural is connected to the all neurons  of the previous layer  
  
- ReLU: Transforms a neuron's output by outputting the input directly if it is positive, and returning zero for all negative values.

- Embedding layer: Takes an index assigned to each word from the vocabulary, and maps it to a representation of that word with a determined dimension.
- Mean layer: A layer that takes the mean of each feature from the embedding, and it output the same number of features as teh embedding sites. It doesn't have any trainable parameters.

## Traditional language models

### N-grams

An n-gram is a sequence of n consecutive words (or characters) from a text. The idea is that you can estimate the probability of a word based on the words that came before it.  

$$P(w2|w1) = \frac{\text{count(w1,w2)}}{\text{count(w1)}} → Bigrams$$  
$$P(w3|w1,w2) = \frac{\text{count(w1,w2,w3)}}{\text{count(w1,w2)}} → Trigrams$$  
$$P(w1,w2,w3) = P(w1) \times P(w2|w1) \times P(w3|w2)$$  

![alt text](image-1.png)  

#### Limitations of N-gram

- Large N-grams needed to capture dependencies between distant words.
- Exponential growth: Increasing n gives better context byt explodes the vocabulary size which needs a lot of space and RAM.
- Sparsity problem: most word combinations never appear in training, its count is 0 making P = 0. The model assigns zero probability to valid sentences it just never saw it. Smoothing techniques (like Laplace smoothing) patch this but don't fully solve it.
- Fixed context window: an n-gram only looks back n-1 words. A bigram is completely blind to anything beyond the previous word.
- No semantic understanding: n-grams are pure statistics over word sequences. "Dog bites man" and "canine attacks person" are completely different patterns to an n-gram model, even though they mean the same thing.  

---

### Recurrent neural networks (RNN)

An RNN process a sequence one token at time, maintaining a hidden state that gets updated at each step. The key idea: the hidden state is a form of memory, it carries information from previous steps into the current one.  

At each step $t$:

- Input: current word $x_t$ (as vector/embedding)
- previous hidden state:  $h_{t-1}$
- New hidden state $h_t = tanh (W_x \cdot X_t + W_h \cdot h_{t-1} + b)$
- Output $y_t = W_{\gamma} \cdot h_t$  

The same weight matrices ($W_x W_h W_{\gamma}$) are reused at every step, that's what "recurrent" means

![alt text](image-2.png)

#### Why is RNN better than n-grams (what is solved)

- **Unlimited context window**: the hidden state theoretically carries information form the very first token all the way to the current one. There is no fixed n; every previous word influences the current hidden state.

- **No sparsity problem**: instead of counting raw co-occurrences, the RNN learns continuous vector representations (embeddings). Two words that appear in similar contexts get similar embeddings, so the model generalizes across unseen combinations naturally.
  
- **Semantic compression**: the hidden state is a dense vector that compresses *meaning*, not just word identity. Related sentences get similar hidden states even if they use different words.

- **No exponential storage growth**: the model parameters are fixed regardless of sequence length. The RNN doesn't need a lookup table that grows with the vocabulary size the way n-gram count tables do

![alt text](image-3.png)

#### Limitations of RNN

- **Vanishing gradient**: this is the biggest one. During back propagation through time (BBTT), gradients gut multiplied together at every step. When those values are < 1 (which they often are after a tanh activation), multiplying them across 50+ steps makes the gradient shrink to essentially zero. The early lyers in the sequence stop learning. In practice, this means an RNN effectively forgets what happened more than ~ 10-20 steps ago, despite theoretically having unlimited context.

- **Exploding gradient**: the opposite can also happen; gradients grow uncontrollably large causing NaN values amd training collapse. Gradient clipping is used to manage this.  

- **sequential computation**: the RNN process tokens one at time in order, because step t depends on step t-1. This makes it impossible to parallelize across a sequence, which means it's slow to train on modern GPW hardware that thrives on parallel operations.

- **Long-range dependencies**: as a practical consequence of vanishing gradients, RNNs struggle with sentences like  "The trophy didn't fit in the suitcase because it was too big." Resolving what "it" refers to requires connecting tokens far apart, and a vanilla RNN often fails this

**LSTMs** and **GRUs** were invented specially to address the vanishing gradient problem by introducing gating mechanisms that let the network learn *what to remember and what to forget*. They are the natural evolution from vanilla RNNs
