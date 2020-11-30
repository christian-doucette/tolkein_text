from collections import Counter

def get_vocab(corpus, min_occurences):
    vocab = Counter()
    for word in corpus:
        vocab[word] += 1

    vocab_top = Counter({k: c for k, c in vocab.items() if c >= min_occurences})
    vocab_tuples = vocab_top.most_common(len(vocab_top))

    word_to_id = Counter({word: i+1 for i,(word, c) in enumerate(vocab_tuples)})

    id_to_word = [word for word, index in word_to_id.items()]
    id_to_word.insert(0, "_")

    return word_to_id, id_to_word





def encode_words(text, word_to_id_dict):
    return [word_to_id_dict[word] for word in text]
