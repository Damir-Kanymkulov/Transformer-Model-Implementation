# Practical 3

import re
import collections

# Get frequency of pairs of characters
def get_stats(vocab):
    pairs = collections.defaultdict(int) 
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

# Merge the most frequent pair into a single token
def merge_vocab(pair, v_in): 
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in: 
        w_out = p.sub(''.join(pair), word)  # Replace the pair with a single token
        v_out[w_out] = v_in[word]  # Update the vocabulary
    return v_out

# Create the initial vocabulary
def preprocess_corpus(corpus):
    vocab = collections.defaultdict(int)
    for sentence in corpus:
        sentence = sentence.strip().lower()  # Lowercase
        sentence = sentence + " </w>"  # Add the word boundary symbol
        words = sentence.split()
        for word in words:
            vocab[' '.join(list(word))] += 1  # Count word frequency
    return vocab

corpus = [
    "Machine learning helps in understanding complex patterns.",
    "Learning machine languages can be complex yet rewarding.",
    "Natural language processing unlocks valuable insights from data.",
    "Processing language naturally is a valuable skill in machine learning.",
    "Understanding natural language is crucial in machine learning."
]
num_merges = 10
vocab_size = 64
vocab = preprocess_corpus(corpus)

for i in range(num_merges):
    pairs = get_stats(vocab)  # Get the frequency of pairs
    best = max(pairs, key=pairs.get)  # Find the most frequent pair
    vocab = merge_vocab(best, vocab)  # Merge the pair into the vocabulary
    if len(vocab) >= vocab_size:
        break  # Stop if the desired vocabulary size is reached

# Function to tokenize sentence
def custom_bpe_tokenizer(sentence, vocab):
    sentence = sentence.strip().lower()  # Lowercase the sentence
    sentence = sentence + " </w>"  # Add the word boundary symbol
    words = sentence.split()
    tokens = []

    for word in words:
        word_tokens = list(word)
        while True:
            pairs = get_stats(dict([(word, 1) for word in word_tokens]))  # Get pairs in the word
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            word_tokens = merge_vocab(best_pair, {word: 1})  # Merge the best pair in the word
        tokens.append(''.join(word_tokens))
    return tokens

# Test the custom tokenizer
custom_tokens = custom_bpe_tokenizer("Machine learning is a subset of artificial intelligence.", vocab)
print("Custom BPE Tokenizer Output:")
print(custom_tokens)

print("\nFinal vocabulary:")
for word, freq in vocab.items():
    print(f"{word}: {freq}")