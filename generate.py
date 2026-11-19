import torch
import v2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
checkpoint = torch.load(
    'shakespeare_gpt.pth',
    map_location=device,
    weights_only=True
)

# Restore the configuration in v2
v2.n_embd = checkpoint['n_embd']
v2.block_size = checkpoint['block_size']
v2.n_head = checkpoint['n_head']
v2.n_layer = checkpoint['n_layer']
v2.dropout = checkpoint['dropout']
v2.vocab_size = len(checkpoint['chars'])

# Reconstruct tokenizer
chars = checkpoint['chars']
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Instantiate model using v2's configuration
model = v2.BigramLanguageModel().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate
prompt = "ROMEO:"
context = torch.tensor(
    [[stoi[c] for c in prompt]],
    dtype=torch.long,
    device=device
)

generated = model.generate(context, max_new_tokens=10000)
generated_text = decode(generated[0].tolist())

with open('more.txt','w',encoding='utf-8') as f:
    f.write(generated_text)