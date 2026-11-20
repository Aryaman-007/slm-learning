import torch
import model
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[i] for i in l])
checkpoint = torch.load(
    'shakespeare_gpt.pth',
    map_location=device,
    weights_only=True
)

# Restore the configuration in model
model.n_embd = checkpoint['n_embd']
model.block_size = checkpoint['block_size']
model.n_head = checkpoint['n_head']
model.n_layer = checkpoint['n_layer']
model.dropout = checkpoint['dropout']
model.vocab_size = checkpoint['vocab_size']

# Reconstruct tokenizer
# chars = checkpoint['chars']
# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for i, ch in enumerate(chars)}

# Instantiate model using model's configuration
model = model.BigramLanguageModel().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate
prompt = "ROMEO:"
context = torch.tensor(
    [tokenizer.encode(prompt).ids],
    dtype=torch.long,
    device=device
)

generated = model.generate(context, max_new_tokens=1000)
generated_text = tokenizer.decode(generated[0].tolist())

with open(r'slm-learning\generated.txt','w',encoding='utf-8') as f:
    f.write(generated_text)