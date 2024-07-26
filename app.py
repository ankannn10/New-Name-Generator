from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F

app = Flask(__name__)
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]

class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def state_dict(self):
        return {f'layer_{i}': [param.clone() for param in layer.parameters()] for i, layer in enumerate(self.layers)}
  
    def load_state_dict(self, state_dict):
        for i, layer in enumerate(self.layers):
            params = state_dict[f'layer_{i}']
            for param, val in zip(layer.parameters(), params):
                param.data.copy_(val)

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False

vocab_size = 27
itos = {i: s for i, s in enumerate("abcdefghijklmnopqrstuvwxyz.")}

torch.manual_seed(10)
n_embd = 32
n_hidden = 512
model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])

# Load the pre-trained model parameters
state_dict = torch.load('model.pth')
#print(state_dict)
model.load_state_dict(state_dict)
model.eval()

#app = Flask(__name__)

def generate_nickname(model, num_samples=20):
    """Generate a list of random nicknames."""
    nicknames = []
    block_size = 8

    for _ in range(num_samples):
        out = []
        context = [0] * block_size  # Initialize context with all dots (or zeros)

        while True:
            # Forward pass the neural net
            context_tensor = torch.tensor([context], dtype=torch.long)
            logits = model(context_tensor)
            probs = F.softmax(logits, dim=1)  # Change dim to 2 for correct softmax
            # Sample from the distribution
            ix = torch.multinomial(probs[0], num_samples=1).item()  # Sample from last output
            # Shift the context window and track the samples
            context = context[1:] + [ix]
            out.append(ix)
            # If we sample the special '.' token, break
            if ix == 0:
                break

        nickname = ''.join(itos[i] for i in out if i != 0)
        nicknames.append(nickname)

    return nicknames

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Generate nicknames when a POST request is made
        num_samples = int(request.form.get('num_samples', 20))
        nicknames = generate_nickname(model, num_samples)
        return jsonify(nicknames)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


'''@app.route('/', methods=['GET', 'POST'])
def generate_nicknames():
    if request.method == 'POST':
        num_samples = int(request.form.get('num_samples', 20))
        nicknames = []
        for _ in range(num_samples):
            out = []
            context = [0] * 8
            while True:
                logits = model(torch.tensor([context]))
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs[0], num_samples=1).item()
                context = context[1:] + [ix]
                if ix == 0:
                    break
                out.append(itos[ix])
            nicknames.append(''.join(out))
        return render_template('index.html', nicknames=nicknames)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)'''
