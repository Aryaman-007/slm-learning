from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# with open(r"slm-learning\TinyStoriesV2-GPT4-train.txt",'r',encoding = 'utf-8') as f:
#     train_text = f.read()

# n = int(0.9*len(text))
# train_text = text[:n]

tokenizer = Tokenizer(BPE(unk_token = "<|unk|>"))

tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size = 2000,
    min_frequency = 2,
    special_tokens = ["<|unk|>","<|endoftext|>"],
    initial_alphabet = ByteLevel.alphabet()
)

train_file = r"slm-learning\TinyStoriesV2-GPT4-train.txt"
tokenizer.train([train_file],trainer)
# tokenizer.train_from_iterator([train_text],trainer = trainer)

tokenizer.save("tinystories_tokenizer.json")

print("Vocabulary size ", tokenizer.get_vocab_size())

samples = [
    "Shakespeare",
    "ROMEO: What is happening?",
    "To be, or not to be",
    "The quick brown fox jumps over the lazy dog."
]

for sample in samples:
    encoded = tokenizer.encode(sample)

    print("\nOriginal:", sample)
    print("Tokens:", encoded.tokens)
    print("Token IDs:", encoded.ids)
    print("Number of tokens:", len(encoded.ids))