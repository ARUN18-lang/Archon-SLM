import torch
import torch.nn.functional as F
import tiktoken
from archon.model import ArchonModel
from archon.config import load_config

device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

model = ArchonModel(load_config())
state_dict = torch.load('Neuraxis.pt', map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


def generate_text(
    prompt, 
    max_new_tokens=200, 
    temperature=0.8, 
    top_k=50
):
    context = (torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(dim=0)).to(device)
    with torch.no_grad():
        out = model.generate(
            context, 
            max_new_tokens=max_new_tokens,
            temperature=temperature, 
            top_k=top_k
        )
    
    return enc.decode(out.squeeze().tolist())



if __name__ == "__main__":
    prompt = "A little girl went to the woods"
    print(generate_text(prompt))
