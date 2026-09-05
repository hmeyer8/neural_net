# Engineer Journal

Newest first.

---

## 9/4 — Starting the agent track, and the keepdim bug wearing a different costume

Big day. New track, three bugs, and one of them is the same bug I wrote about on 7/25 in a completely different outfit.

**Why I'm switching tracks.** The CV work is paused, and I want to be honest in writing about why rather than dressing it up. The stated reason on 8/13 was that I wanted to take a model from raw data to something a person could call. That's still true. The actual reason is that the job I want is applied LLM systems — agents, retrieval, evaluation, the operations around models — and building a fourth CNN would have been comfortable and off-target. The drills stay (`reps.py` runs every morning, that's not negotiable), but the project work moves.

What survives intact is the standard: a thing is done when there's a number, a baseline, and a written account of how it fails. That was the best idea in the CV track and it transfers without modification.

**Why RFCs.** I needed a corpus where I could grade answers instead of eyeballing them. The thing that sold me is this: RFC 2616 §14.9 reads perfectly, quotes cleanly, and was obsoleted in 2014. A retrieval system finds it, cites it correctly, and is wrong — and *nothing about the answer looks wrong*. The retrieval worked. The quote is accurate. The citation points at real text.

You cannot detect that with similarity search, because "is this still in force" isn't in the passage. It's in the document's `Obsoletes`/`Obsoleted-By` graph, which is structured metadata the retriever never consults. So the system needs a second, different kind of lookup, and *that's* the difference between a RAG demo and an agent. The multi-step part isn't decoration — take it out and the failure mode is silent, confident, checkable errors.

The sentence I want to be able to say out loud: **the hard problem in a normative corpus isn't finding the text, it's knowing whether the text you found is still in force.** That's also true of policy, regulation, and procedure, which is the point.

**The keepdim bug, again.** Here's the one I want to remember.

The corpus fetcher pulls ~9,800 documents. The first run died partway through on a dropped connection. Fixed that. Reran it. It printed `2895 documents`, exited 0, and took about two seconds.

It had fetched nothing. The resume guard was `if existing > 1000: return existing` — the assumption being that a directory with a lot of files in it is a finished corpus. It isn't. It was 2,895 of 9,835, and the function looked at a populated directory and declared victory.

Now hold that next to 7/25. The `keepdim` bug was shape-compatible, so torch ran it without a single error and the rows just didn't sum to 1. This one is *count-compatible* — a corpus with files in it looks like a corpus — so the fetcher ran without a single error and the corpus was 29% complete. Same class of failure. Both run clean. Both produce something that has the right shape and the wrong contents.

And this one is worse, which took me a minute to see. The `keepdim` bug produced wrong numbers immediately, right there in the notebook. This one produces *plausible* numbers much later and somewhere else entirely: I'd have built the index, run retrieval, and gotten a recall@5 on 29% of the corpus. Recall@5 on a third of a corpus looks exactly like recall@5 on all of it. There's no shape to check, no assertion that fires, no row that fails to sum to 1. It's just quietly a different experiment than the one I'd have written down in the ledger.

The only reason I caught it: **it finished too fast.** That's the entire detection mechanism. Not a test, not an assertion — a feeling about elapsed time. That's not good enough, and the fix is the rule I'm writing into the playbook: *check the quantity produced against what you expected, not the exit code.* A run that finishes suspiciously fast did.

Resume is a set difference now — index numbers, minus what's on disk, minus what's known to 404 — and it prints all three counts, so "complete" is a claim with evidence behind it instead of an inference from a file count.

**Shape versus sequence.** I'd written a line in the parser docstring that I was pleased with: *a parser you have not tested against 1989 is a parser that works on 2014.* Then the corpus landed and I actually tested it against 1989. Half right.

The heading detector is a **shape** test — column 0, leading integer, some text. Ordinary prose matches that shape. RFC 2626 is a Y2K survey document full of bare years, and it produced **550 phantom sections** from lines reading `2000  found at line 3182:`. RFC 1035 wrapped a sentence onto a line starting `25 (SMTP).  If this bit is set…` and that became section 25.

The fix is the useful idea: section numbers aren't just a shape, they're a **sequence**. A top-level number never jumps more than one past the highest you've already seen. `2000` after section 6 is not a section. That single constraint drops RFC 2626 from 629 sections to 78 and removes nothing from RFC 7234, 2616, 8446, 9110, 793, or the ~997-section NFS specs.

**The metric that moved the wrong way and was right.** Adding that guard raised "documents parsing to one section" from 581 to 641. My first read was that I'd broken 60 documents. I hadn't — those are early-70s memos where the rejected "headings" were host tables (`65  UCLA  IBM-360/91`), street addresses (`1400 Wilson Boulevard`), and wrapped prose. They had *fake* structure before and correctly have none now.

Worth sitting with, because I'd have accepted the opposite conclusion if I hadn't looked: a number moving in the bad direction can be the number getting more honest. Same reason I check what's *behind* a metric before believing it went up.

The real finding underneath: 641 documents have no numbered sections at all, and every one of them is pre-1995. Zero after. That's a property of the format, not a parser failure — but it's a genuine limitation for retrieval, since those can't be chunked below document level, and it goes on the limitations page rather than being quietly ignored.

**GPU.** Moved torch to the cu130 wheel for the 3050. Encoding goes 412 → 2,988 texts/s, 7.3×. What matters isn't the number, it's that a full-corpus index build drops from ~20 minutes to ~2, which is what makes running the chunking ablation *twice* affordable instead of once-and-hoping.

One decision I want to remember the reasoning for: the device is deliberately **not** in the config hash. recall@5 is the same number on a GPU or a CPU, so folding the device into the identity of a configuration would split the ledger into two families of runs that aren't actually comparing anything different. It gets its own column instead — quality keyed by the hash, latency keyed by the hash *and* the device. Also learned that a seed isn't reproducibility on CUDA: cuDNN benchmarks algorithms at runtime, so identically seeded runs can differ in the last decimals, which is enough to flip two near-tied retrieval hits and read as a regression that never happened.

**What I'm taking from today.** Every real bug today was silent. None of them threw. The connection drop was the only one that announced itself, and it was the least dangerous of the three.

I think the actual skill I'm building isn't writing the retrieval system — it's developing the instinct for *where a system can lie to you*. Shape-compatible broadcasts. Count-compatible corpora. Prose that matches a heading regex. Exit code 0. In every case the code is fine and the assumption underneath it is wrong, and there's nothing to read in the traceback because there is no traceback.

Next: BM25 by hand, then chunking measured two ways instead of picked one way, then the first real recall@5.

---

## 8/19: reps.py notes
Four axes for a batch of images: x.shape = (N, C, H, W). N is number of image in batch, C is color channel- RGB usually. H is pixel row, w is pixel column. 

Today also covered .mean, .sum, .shape and .max functions. Main rule is that whatever axis you name in axis = ... gets collapsed and disappears. Everything else stays the same, in order

On a (3,5), 2D arraynamed x, x.mean(axis = (0)) collapses the rows and gives the per column mean, leaving it with x.mean(axis = (0)).shape = (5,)-- NOT (0,1) - a 0 in a shape means an axis that exists with zero elements, not an axis that got removed

keepdims = True, it does the same thing but collapses the axis to 1. so x.mean(axis = (0), keepdims = True).shape = (1,5). This is what lets x - mean broadcast cleanly without reshaping by hand

## 8/13 — Pivoting to computer vision

Stepping off Zero to Hero and into vision. Two reasons.

First, the marginal return on more Karpathy has dropped. I understand the engine now — a `Value` graph, local derivatives, chain rule, `+=` accumulating over multiple paths. Watching another lecture is not going to teach me something the micrograd build didn't. What I have not done is take a model from raw data all the way to something a person could actually call. That's the gap, and it's a bigger one than any remaining lecture closes.

Second, the `keepdim` bug from 7/25 has been sitting with me. Not because the concept was hard — I knew exactly what broadcasting does, and I could derive why `[27]` and `[27, 1]` behave differently at a whiteboard. The problem was that I didn't *catch* it, because it ran clean. That's not a math gap. That's a reps gap with the library. I've written more derivations than NumPy this year and it shows.

So the plan is different from how I'd have written it six months ago. It doesn't re-teach me backprop. It drills the API — `axis` vs `dim`, `view` vs `reshape`, when `keepdim` is load-bearing — and then spends the real time on the things I've never actually done: building a dataset with written labeling rules, running controlled experiments, doing an honest failure analysis, and shipping the thing behind an endpoint in a container.

Full plan in [ROADMAP.md](ROADMAP.md). Eight weeks, three hours a day, six days a week with the seventh as flex. I'm writing the flex day in on purpose. I train for an Ironman on the same calendar and I've never seen a training block survive without recovery built in — I don't know why I'd expect a study block to be different.

The thing I'm actually optimizing for: at the end of this I want someone to hand me a blank repo and an ambiguous image problem, and I start asking the right questions instead of waiting to be told the next step.

---

## 7/25 — Sampling, and the broadcasting bug I nearly shipped

**Sampling from a distribution.** Before touching the real model I played with `torch.multinomial` on a toy example — a random 3-element vector, normalized into probabilities, then sampled 20 times with replacement. It's the exact machinery I need for names: give it a distribution, it hands back an index proportional to that distribution's weights. Applied to the real thing: `p = N[0].float(); p = p/p.sum()`, feed `p` into `multinomial`, get back an index, look it up in `itos`. That's one letter generated from nothing but the empirical frequency of what usually starts a name.

**Scaling that to the whole matrix — and a bug I nearly shipped.** Doing `p = N[0].float() / N[0].float().sum()` one row at a time works, but the real move is normalizing the entire `N` matrix at once: `P = N.float()`, then divide every row by its own row-sum, in one shot, no loop. That's `P.sum(1, keepdim=True)`.

I almost left `keepdim` off, and it would've run without complaining — which is exactly what makes it dangerous. Here's what's actually happening. `P.sum(1)` collapses the row dimension, so instead of a `[27, 1]` column it hands back a flat `[27]` vector. When that gets broadcast against `P` for the division, PyTorch lines up shapes from the trailing dimension and silently prepends a 1 to the shorter one — so `[27]` becomes `[1, 27]`. A `[1, 27]` divisor broadcasts *across rows*, meaning entry `j` of that vector gets applied as the divisor for column `j`, in every row. But entry `j` was never "the total for column `j`" — it was "the total for row `j`." So you're dividing each column by a number that has nothing to do with that column. It's not a clean row/column swap, it's just an index mismatch that happens to be shape-compatible, so torch runs it without a single error. The rows don't sum to 1. The columns don't either. It's just wrong, quietly.

`keepdim=True` fixes it by keeping that dimension as a `1` instead of dropping it — `[27, 1]`. Now broadcasting stretches that single column value across every entry *in its own row*, which is what you actually want: each row's own total gets divided out of itself, and the row becomes a real probability distribution.

Lesson I'm taking from this: `.sum()`'s default behavior of squeezing out the reduced dimension is a footgun the moment you're about to broadcast the result back against the original tensor. `keepdim=True` isn't stylistic — it's what pins the divisor to the axis you actually meant instead of letting broadcasting guess for you.

**First names out of the model.** With `P` built and every row properly normalized, the generation loop is dead simple: start at index 0 (the `.` token), sample the next character from `P[ix]`, move to that index, repeat until you land back on `.`. 50 samples in and — they're bad. Recognizably name-*shaped* (right letter frequencies, plausible lengths) but not real names, because a bigram only ever looks one character back. No memory of anything before that. Next step is presumably going to be about giving the model more context than a single trailing character.

---

## 7/17 — makemore, bigrams

I'll be posting more now — I'm shooting to finish Andrej's ZTH lectures by the end of July. Heavy hours, but I want the fundamentals nailed down before I start Stanford's CS231n course. Going for consistency even in an environment that doesn't support studying, which is exactly where I'm at right now.

The overall goal of makemore: feed it a big list of names (`names.txt`, ~32k of them) and have it invent *new* names that sound name-like. The whole first pass does this with nothing fancier than counting — a bigram model.

**Bigrams.** A bigram is just two characters in a row. The idea is that if you know the current letter, you can guess the next one by looking at what usually follows it in real names. So the model only ever looks one character back. First I built this as a Python dict: walk every word, pad it with a start/end marker, `zip(chs, chs[1:])` to get consecutive pairs, and tally each pair in `b`. Sorting by count shows the common ones jump out — stuff like `n` at the end of a word, `a` after lots of letters, etc.

**Why a 2D array beats the dict.** The dict works but it's clunky to compute with. Way better to store the same info in a 27x27 tensor `N`, where the row is the first character, the column is the second, and the entry is how many times that pair showed up. 27 because 26 letters plus one special `.` token that marks both the start and end of a word (Andrej collapses start/end into a single character instead of separate `<S>`/`<E>` — cleaner).

**The stoi / itos trick.** Tensors are indexed by integers, but our data is characters, so we need a lookup table both directions. `stoi` maps character → integer (`.` = 0, a=1, b=2 ...), `itos` is the reverse. This is the little bridge between "human" data and something torch can index into. Then the fill loop is the same bigram walk as before, except now instead of a dict key it's `N[ix1, ix2] += 1`.

**Visualizing it.** The matplotlib cell is the payoff — you plot `N` as an image with each cell labeled by its bigram and its count. You can literally *see* the structure: the `.` row (word starts) lights up on common first letters, the `.` column (word ends) lights up on common last letters, and dead combinations are near zero. The model is just this picture.

**From counts to probabilities.** Last thing today: `p = N[0].float(); p = p/p.sum()`. Row 0 is the `.` row — the counts of every letter that *starts* a word. Divide by the sum and it becomes a probability distribution: "given we're at the start, here's how likely each first letter is." That's the actual model. Next step is to sample from these rows to generate names.

Side note — Andrej's for loops that chew through the data are so clean they got me fired up about algorithms, so I'm now doing one LeetCode problem a day alongside this.

---

## 6/23 — nn.py

Now that the engine (Value + backprop) works, `nn.py` builds an actual neural net on top of it. The whole file is just three ideas stacked on each other: Neuron → Layer → MLP.

**Neuron.** Andrej starts with the biological picture: a neuron gets a bunch of inputs `x`, each flowing in along a synapse that has a weight `w`. The weight is the *strength* of that connection. So the neuron multiplies each input by its weight, sums them all up, and adds a bias `b` (the bias is basically how trigger-happy the neuron is overall — how easy it is to make it fire regardless of input). That's just the dot product w·x + b. Then it runs that through a nonlinearity (`relu` in the final code, `tanh` in the video) to squash it. In code the weights are random `Value`s, the bias starts at 0, and `__call__` does exactly `sum(wi*xi) + b` then the activation. Key thing: every `w` and `b` is a `Value`, so backprop already knows how to differentiate through all of it.

**Layer.** A layer is just a list of neurons that all look at the *same* input independently — they're not wired to each other. So if you want 4 outputs, you make 4 neurons, feed each the same `x`, and collect their outputs into a list. That's the whole class.

**MLP.** Multi-layer perceptron = layers stacked in sequence. The output of one layer becomes the input to the next. `__call__` literally just loops: `for layer in self.layers: x = layer(x)`. You define it with sizes like `MLP(3, [4, 4, 1])` = 3 inputs, two hidden layers of 4, one output.

**parameters().** Every class has this and it just gathers up all the `Value`s (all the weights and biases) into one flat list. This is what you need for training — these are the knobs gradient descent turns. `MLP.parameters()` reaches down through every layer and every neuron to grab them all.

**zero_grad().** The one that bit Andrej in the video: before each backward pass you have to reset every parameter's `.grad` to 0. Since backprop uses `+=`, if you don't zero first you keep accumulating gradients from previous passes and your training goes sideways. Lives in the `Module` base class so everything inherits it.

That's it — the "neural net" is nothing more than a big pile of `Value`s doing multiply/add/relu, and because the engine already handles the gradients, training just falls out of it.

---

## 5/31 — Micrograd

This has so far taught me that backpropagation is the gradient of a node WRT the final value (call it L). By convention each node's `.grad` holds ∂L/∂node, so `out.grad` is ∂L/∂out and I'm solving for `self.grad` = ∂L/∂self.

Chain rule says route L's sensitivity through out:

`self.grad += out.grad * (∂out/∂self)`

That's the whole template — every op just swaps in a different local derivative ∂out/∂self.

**Addition:** out = self + other, so ∂out/∂self = 1 → `self.grad += out.grad * 1.0`. Addition just passes the upstream gradient through untouched.

**Multiplication:** out = self * other, so ∂out/∂self = other → `self.grad += out.grad * other.data`. It's `other.data` (the value) not `other.grad`, because the local derivative is the other operand itself. Nudge self by ε and the product moves by other·ε.

The `+=` matters: if a node feeds multiple downstream ops, L depends on it through several paths and you sum over all of them (multivariable chain rule). That's also why grads get zeroed before each pass. The `a + a` case proves it — correct grad is 2, and `+=` gets it right where `=` wouldn't.

Andrej (yep we're on a first name basis after how many hours I'll be spending on this series) explained it very well.