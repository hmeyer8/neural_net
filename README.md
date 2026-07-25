##### Micrograd

5/31 This has so far taught me that backpropagation is the gradient of a node WRT the final value (call it L). By convention each node's `.grad` holds ∂L/∂node, so `out.grad` is ∂L/∂out and I'm solving for `self.grad` = ∂L/∂self.

Chain rule says route L's sensitivity through out:

`self.grad += out.grad * (∂out/∂self)`

That's the whole template — every op just swaps in a different local derivative ∂out/∂self.

**Addition:** out = self + other, so ∂out/∂self = 1 → `self.grad += out.grad * 1.0`. Addition just passes the upstream gradient through untouched.

**Multiplication:** out = self * other, so ∂out/∂self = other → `self.grad += out.grad * other.data`. It's `other.data` (the value) not `other.grad`, because the local derivative is the other operand itself. Nudge self by ε and the product moves by other·ε.

The `+=` matters: if a node feeds multiple downstream ops, L depends on it through several paths and you sum over all of them (multivariable chain rule). That's also why grads get zeroed before each pass. The `a + a` case proves it — correct grad is 2, and `+=` gets it right where `=` wouldn't.

Andrej (yep we're on a first name basis after how many hours I'll be spending on this series) explained it very well.


##### nn.py

6/23 Now that the engine (Value + backprop) works, `nn.py` builds an actual neural net on top of it. The whole file is just three ideas stacked on each other: Neuron → Layer → MLP.

**Neuron.** Andrej starts with the biological picture: a neuron gets a bunch of inputs `x`, each flowing in along a synapse that has a weight `w`. The weight is the *strength* of that connection. So the neuron multiplies each input by its weight, sums them all up, and adds a bias `b` (the bias is basically how trigger-happy the neuron is overall — how easy it is to make it fire regardless of input). That's just the dot product w·x + b. Then it runs that through a nonlinearity (`relu` in the final code, `tanh` in the video) to squash it. In code the weights are random `Value`s, the bias starts at 0, and `__call__` does exactly `sum(wi*xi) + b` then the activation. Key thing: every `w` and `b` is a `Value`, so backprop already knows how to differentiate through all of it.

**Layer.** A layer is just a list of neurons that all look at the *same* input independently — they're not wired to each other. So if you want 4 outputs, you make 4 neurons, feed each the same `x`, and collect their outputs into a list. That's the whole class.

**MLP.** Multi-layer perceptron = layers stacked in sequence. The output of one layer becomes the input to the next. `__call__` literally just loops: `for layer in self.layers: x = layer(x)`. You define it with sizes like `MLP(3, [4, 4, 1])` = 3 inputs, two hidden layers of 4, one output.

**parameters().** Every class has this and it just gathers up all the `Value`s (all the weights and biases) into one flat list. This is what you need for training — these are the knobs gradient descent turns. `MLP.parameters()` reaches down through every layer and every neuron to grab them all.

**zero_grad().** The one that bit Andrej in the video: before each backward pass you have to reset every parameter's `.grad` to 0. Since backprop uses `+=`, if you don't zero first you keep accumulating gradients from previous passes and your training goes sideways. Lives in the `Module` base class so everything inherits it.

That's it — the "neural net" is nothing more than a big pile of `Value`s doing multiply/add/relu, and because the engine already handles the gradients, training just falls out of it.


##### makemore

7/17 I'll be posting more now — I'm shooting to finish Andrej's ZTH lectures by the end of July. Heavy hours, but I want the fundamentals nailed down before I start stanfords CS 231n course. Going for consistency even in an environment that doesn't support studying, which is exactly where I'm at right now.

The overall goal of makemore: feed it a big list of names (`names.txt`, ~32k of them) and have it invent *new* names that sound name-like. The whole first pass does this with nothing fancier than counting — a bigram model.

**Bigrams.** A bigram is just two characters in a row. The idea is that if you know the current letter, you can guess the next one by looking at what usually follows it in real names. So the model only ever looks one character back. First I built this as a Python dict: walk every word, pad it with a start/end marker, `zip(chs, chs[1:])` to get consecutive pairs, and tally each pair in `b`. Sorting by count shows the common ones jump out — stuff like `n` at the end of a word, `a` after lots of letters, etc.

**Why a 2D array beats the dict.** The dict works but it's clunky to compute with. Way better to store the same info in a 27x27 tensor `N`, where the row is the first character, the column is the second, and the entry is how many times that pair showed up. 27 because 26 letters plus one special `.` token that marks both the start and end of a word (Andrej collapses start/end into a single character instead of separate `<S>`/`<E>` — cleaner).

**The stoi / itos trick.** Tensors are indexed by integers, but our data is characters, so we need a lookup table both directions. `stoi` maps character → integer (`.` = 0, a=1, b=2 ...), `itos` is the reverse. This is the little bridge between "human" data and something torch can index into. Then the fill loop is the same bigram walk as before, except now instead of a dict key it's `N[ix1, ix2] += 1`.

**Visualizing it.** The matplotlib cell is the payoff — you plot `N` as an image with each cell labeled by its bigram and its count. You can literally *see* the structure: the `.` row (word starts) lights up on common first letters, the `.` column (word ends) lights up on common last letters, and dead combinations are near zero. The model is just this picture.

**From counts to probabilities.** Last thing today: `p = N[0].float(); p = p/p.sum()`. Row 0 is the `.` row — the counts of every letter that *starts* a word. Divide by the sum and it becomes a probability distribution: "given we're at the start, here's how likely each first letter is." That's the actual model. Next step is to sample from these rows to generate names.

Side note — Andrej's for loops that chew through the data are so clean they got me fired up about algorithms, so I'm now doing one LeetCode problem a day alongside this.

7/25 **Sampling from a distribution.** Before touching the real model I played with `torch.multinomial` on a toy example — a random 3-element vector, normalized into probabilities, then sampled 20 times with replacement. It's the exact machinery I need for names: give it a distribution, it hands back an index proportional to that distribution's weights. Applied to the real thing: `p = N[0].float(); p = p/p.sum()`, feed `p` into `multinomial`, get back an index, look it up in `itos`. That's one letter generated from nothing but the empirical frequency of what usually starts a name.

**Scaling that to the whole matrix — and a bug I nearly shipped.** Doing `p = N[0].float() / N[0].float().sum()` one row at a time works, but the real move is normalizing the entire `N` matrix at once: `P = N.float()`, then divide every row by its own row-sum, in one shot, no loop. That's `P.sum(1, keepdim=True)`.

I almost left `keepdim` off, and it would've run without complaining — which is exactly what makes it dangerous. Here's what's actually happening. `P.sum(1)` collapses the row dimension, so instead of a `[27, 1]` column it hands back a flat `[27]` vector. When that gets broadcast against `P` for the division, PyTorch lines up shapes from the trailing dimension and silently prepends a 1 to the shorter one — so `[27]` becomes `[1, 27]`. A `[1, 27]` divisor broadcasts *across rows*, meaning entry `j` of that vector gets applied as the divisor for column `j`, in every row. But entry `j` was never "the total for column `j`" — it was "the total for row `j`." So you're dividing each column by a number that has nothing to do with that column. It's not a clean row/column swap, it's just an index mismatch that happens to be shape-compatible, so torch runs it without a single error. The rows don't sum to 1. The columns don't either. It's just wrong, quietly.

`keepdim=True` fixes it by keeping that dimension as a `1` instead of dropping it — `[27, 1]`. Now broadcasting stretches that single column value across every entry *in its own row*, which is what you actually want: each row's own total gets divided out of itself, and the row becomes a real probability distribution.

Lesson I'm taking from this: `.sum()`'s default behavior of squeezing out the reduced dimension is a footgun the moment you're about to broadcast the result back against the original tensor. `keepdim=True` isn't stylistic — it's what pins the divisor to the axis you actually meant instead of letting broadcasting guess for you.

**First names out of the model.** With `P` built and every row properly normalized, the generation loop is dead simple: start at index 0 (the `.` token), sample the next character from `P[ix]`, move to that index, repeat until you land back on `.`. 50 samples in and — they're bad. Recognizably name-*shaped* (right letter frequencies, plausible lengths) but not real names, because a bigram only ever looks one character back. No memory of anything before that. Next step is presumably going to be about giving the model more context than a single trailing character.