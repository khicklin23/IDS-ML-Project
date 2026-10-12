# How the model actually learns (and what "weights" even are)

This document assumes you know **nothing** about machine learning. If a word is unusual, it gets a plain-English definition in parentheses the first time it shows up. Every number in here is real, pulled straight from this project's code, nothing is a round, made-up "for example" figure.

By the end you'll know:
1. What a neural network is made of, physically (as numbers in a file)
2. What "training" is actually doing to those numbers, step by step
3. What the model looks like once training is done, and how we use it on new data
4. How all of this compares to how something like Claude or ChatGPT is built, same basic idea, wildly different scale

---

## 1. The problem, in one sentence

We have a big spreadsheet of network connections ("flows", one row per connection between two computers, like a phone call record: who called whom, how long, how much data). Some rows are normal traffic, some are attacks. We want a program that looks at a row it's never seen before and guesses which kind of traffic it is.

A neural network is one way to build that guessing program. Here's what it's made of and how it gets good at guessing.

---

## 2. What a neural network actually *is*

Forget the word "neural" for a second, it's a historical name, not a biological brain. A neural network is just a **big stack of adjustable math**. Think of it like a sound mixing board with hundreds of sliders: each slider is a number, you can turn any of them up or down, and the combination of all the slider positions changes the output sound. "Training" is the process of automatically finding slider positions that produce good output.

### The smallest building block: one neuron

A single "neuron" does three boring things to a handful of numbers:

```
  input numbers           weights (adjustable)
  ┌──────────┐            ┌──────────┐
  │ x1 = 0.8 │  ───────▶  │ w1 = 1.2 │
  │ x2 = 0.3 │  ───────▶  │ w2 = -0.5│   →  sum = (x1·w1)+(x2·w2)+bias
  │ x3 = 0.1 │  ───────▶  │ w3 = 2.0 │       = (0.8·1.2)+(0.3·-0.5)+(0.1·2.0) + bias
  └──────────┘            └──────────┘       = 0.96 - 0.15 + 0.20 + bias
                                bias = 0.1    = 1.11
                                               │
                                               ▼
                                     activation function
                                     (squashes the number,
                                      e.g. "if negative, make it 0")
                                               │
                                               ▼
                                          output = 1.11
```

- **Weight**: a number that says "how much does this particular input matter, and in which direction." A big positive weight means "when this input goes up, push my answer up a lot." A negative weight means the opposite.
- **Bias**: one extra adjustable number per neuron, a flat offset, like a thermostat's baseline setting before any input is considered.
- **Activation function**: a simple rule applied after the weighted sum, so the network can represent "this matters only past a certain point" instead of everything being a straight line. This project uses **ReLU** ("if the number is negative, make it zero; otherwise leave it alone"), about the simplest activation function that exists.

That's the entire "neuron." It has no logic, no memory, no understanding, it's a weighted sum plus a simple on/off-ish rule. A **neural network** is just thousands of these wired together in layers, each layer's output feeding the next layer's input.

### Stacking neurons into a network

Here's the actual architecture this project uses for the attack-type classifier (`src/ids_ml/models/classifier.py`), with its real dimensions:

```
INPUT                HIDDEN LAYER 1         HIDDEN LAYER 2       OUTPUT
23 numbers       32 neurons (ReLU)      16 neurons (ReLU)     6 numbers
(one flow,    ┌─○─○─○─○─○─○─○─○─○─┐   ┌─○─○─○─○─○─○─○─○─┐   ┌─○─○─○─○─○─○─┐
 turned into  │ ○─○─○─○─○─○─○─○─○ │   │ ○─○─○─○─○─○─○─○ │   │             │
 numbers) ───▶│ ○─○─○─○─○─○─○─○─○ │──▶│ ○─○─○─○─○─○─○─○ │──▶│ one score   │
              │ ○─○─○─○─○─○─○─○─○ │   └─────────────────┘   │ per attack  │
              └───────────────────┘                          │ type        │
                                                               └─────────────┘
    736 weights + 32 biases     512 weights + 16 biases    96 weights + 6 biases
         (layer 1)                    (layer 2)                 (layer 3)

                        TOTAL: 1,398 adjustable numbers
```

Every line in that diagram is one weight. There is no hidden logic beyond "multiply, add, repeat." The entire "intelligence" of this model is **1,398 numbers**, that's the complete list of things training is allowed to change. (The sequence-aware "v3/v4" model, which reads a whole conversation of flows instead of one at a time, uses a slightly different wiring called a GRU, more on that in [the model-versions README section](../README.md#model-versions-v1-v2-v3-v4), and has 5,670 adjustable numbers. Still tiny.)

### Where does "23 numbers" come from?

A raw flow row looks like this:

```
duration_s: 2.74   src_port: 2373   dst_port: 443   protocol: tcp
service: https     packets_sent: 22   packets_recv: 15
bytes_sent: 12404  bytes_recv: 9751   outcome: timeout   prior_gap_s: 0.0
```

A neural network can only multiply and add numbers, it can't read the word `"tcp"`. So every flow gets converted into a plain list of numbers before it ever touches the network:

- **Numeric fields** (duration, ports, packet/byte counts, timing) get *scaled* so they're all roughly the same size, otherwise "bytes_sent: 12404" would completely drown out "duration_s: 2.74" just because it's a bigger number, for no meaningful reason. This project uses a standard method called a `StandardScaler` (subtract the average, divide by the spread).
- **Category fields** (`protocol`, `service`, `outcome`) get turned into **one-hot encoding**: instead of one slot holding the text "tcp", you get one slot *per possible value*, and exactly one of them is set to 1 while the rest are 0, like a row of light switches where only one bulb is ever on. `protocol` has 2 possible values (tcp, udp) → 2 switches. `service` has 9 (dns, ftp, http, https, ntp, other, rdp, smtp, ssh) → 9 switches. `outcome` has 4 (established, rejected, reset, timeout) → 4 switches. That's 2+9+4 = 15 switches, plus the 8 numeric fields (duration, ports, packet/byte counts, timing gap) = **23 numbers total**, from 11 original fields.

This conversion step (called the **preprocessor**) is itself saved to disk alongside the model, more on that in Part 4, because it turns out to matter a lot.

---

## 3. What "training" actually does, step by step

Training is the process of nudging those 1,398 numbers from random junk into something useful, by repeatedly showing the network examples where we already know the right answer, and adjusting the numbers a tiny bit every time it's wrong.

### The metaphor: an archer with a coach, blindfolded, correcting by feel

Imagine teaching someone archery, except they're blindfolded and can only adjust their aim based on a coach saying "a little left" or "a little higher" after every shot, never telling them *exactly* why, just the direction and roughly how much. Over hundreds of shots, their aim converges on the bullseye, purely through many small corrections. No single correction fixes everything; it's the *accumulation* of thousands of tiny nudges that eventually produces a decent archer.

Neural network training is exactly this, except:
- the "shot" is the network's guess for one flow (e.g., "73% sure this is `beaconing`")
- the "bullseye" is the correct label we already know from the data generator
- the "coach's correction" is a mathematical procedure called **backpropagation**
- the "aim" being adjusted is all 1,398 weights and biases, simultaneously, a little bit each time

### The actual loop

```
                    ┌─────────────────────────────────────────┐
                    │   REPEAT THIS FOR EVERY BATCH OF FLOWS   │
                    └─────────────────────────────────────────┘
                                       │
   1. FORWARD PASS                    ▼
      Feed a batch of flows through the network using its
      CURRENT weights. Get a guess for each one.
      e.g. "flow #4821: 12% benign, 71% beaconing, 17% other..."
                                       │
                                       ▼
   2. LOSS (how wrong was that?)
      Compare the guess to the real answer (the generator's
      ground-truth label). A wrong, confident guess gets
      punished harder than a wrong, unsure guess. This
      project uses "cross-entropy loss" for the multiclass
      model, a standard way to score "how wrong" a probability
      guess was.
                                       │
                                       ▼
   3. BACKPROPAGATION (whose fault was it?)
      Work backwards through the network, layer by layer,
      computing exactly how much EACH of the 1,398 numbers
      contributed to that wrongness. This is pure calculus
      (the chain rule), done automatically by PyTorch, nobody
      hand-derives this.
                                       │
                                       ▼
   4. OPTIMIZER STEP (nudge the dials)
      Adjust every weight a small step in the direction that
      would have reduced the error. This project uses "Adam,"
      a popular, well-behaved nudging rule. The step size is
      the "learning rate" (0.001 here), too big and the dials
      overshoot and never settle; too small and training
      crawls.
                                       │
                                       ▼
          Repeat for every batch of flows in the training set.
          One full pass through ALL training flows = one "epoch."
                                       │
                                       ▼
   5. CHECK PROGRESS ON FLOWS IT WASN'T TRAINED ON
      After each epoch, test the current weights on a separate
      "validation" set the network never got to adjust its
      weights against. If validation performance stops
      improving for 10 epochs in a row ("patience"), STOP,
      this is "early stopping." It exists because a network
      can "memorize" its training examples instead of learning
      the general pattern (called overfitting, like a student
      who memorizes last year's exam answers instead of
      understanding the subject, and then fails this year's
      slightly different exam).
```

That whole 5-step loop is the entirety of "AI training" for this project. There's no reasoning, no planning, no understanding of what "beaconing" *means*, it's gradient-driven trial and error at a scale no human could do by hand, automated into a program that finishes in minutes.

### A concrete pass, with real numbers

One real benign flow, after preprocessing, starts as a list of 23 numbers. The network's first hidden layer turns those 23 numbers into 32 new numbers (via that weighted-sum-plus-ReLU recipe from Part 2), the second hidden layer compresses those 32 numbers down to 16, and the final layer produces 6 raw scores, one per possible label:

```
raw output (logits):  [ 4.1,  -2.3,  -1.8,  -3.0,  -2.9,  -1.1 ]
                      benign  scan   bforce  beacon  flood  outbound

turn into probabilities with "softmax"
(forces them to sum to 100% while keeping the ranking):

                      [ 98.8%, 0.2%,  0.3%,  0.1%,  0.1%,  0.5% ]
                      benign  scan   bforce  beacon  flood  outbound
```

If the true label really was `benign`, the loss is small and the nudge is tiny. If the true label was actually `beaconing` and the network was this confidently wrong, the loss is large, and backpropagation pushes every weight that contributed to that mistake a bit harder than usual, specifically, the nudge is proportional to how wrong and how confident the guess was.

---

## 4. After training: what "the model" physically is, and how it gets used

Training produces exactly three files, saved to a folder (see `ids_ml/models/train.py::save_artifact`). None of them is a program, they're just data:

```
artifact_multiclass/
├── model.pt          ← the 1,398 (or 5,670) learned numbers themselves,
│                        saved by PyTorch. Nothing more. For the sequence
│                        model used in this project's "v4," this file is
│                        about 25 KB.
├── preprocessor.pkl   ← the EXACT scaling/encoding rules fit on the
│                        training data (the averages, spreads, and the
│                        list of category values seen during training).
│                        Needed so brand-new flows get converted into
│                        numbers in EXACTLY the same way the training
│                        data was, every time, forever.
└── metadata.json      ← a small text file describing the shape: how many
                          inputs, how many hidden neurons, what the 6
                          output labels are called, and which training
                          settings were used. Example, straight from this
                          project's v4 model:
                          {
                            "architecture": "GRU (session-level)",
                            "hidden_dim": 32,
                            "class_names": ["benign","port_scan",
                              "auth_bruteforce","beaconing",
                              "service_flood","outbound_transfer"],
                            "train_config": {"lr": 0.001,
                              "max_epochs": 60, "batch_size": 32}
                          }
```

That's it. "The trained model" is a small file of numbers plus a small file of rules for converting raw data into numbers plus a small file describing the shape. Nobody can read `model.pt` and understand *why* it works, it's 1,398 numbers that happen to produce good answers, the same way you can't look at someone's muscle memory and read off the golf swing.

### Using it on new data (inference)

This is the critical difference from training: **once training finishes, the weights are frozen.** Using the model on a brand-new flow is just one single forward pass (step 1 from the training loop) with no loss, no backpropagation, no adjustment, just arithmetic, and it's fast (milliseconds, not minutes):

```
 NEW, NEVER-SEEN FLOW
         │
         ▼
 Load the saved preprocessor.pkl  →  convert the raw flow into
 the same 23 numbers, using the EXACT same scaling rules fit
 during training (ids_ml/inference.py::score_dataset_multiclass)
         │
         ▼
 Load the saved model.pt weights into the same network shape
 described in metadata.json  →  run one forward pass
         │
         ▼
 Output: [96.8% benign, 0.2% port_scan, 0.3% auth_bruteforce,
          0.1% beaconing, 0.1% service_flood, 0.6% outbound]
         │
         ▼
 Take the highest one → "predicted_scenario: benign, confidence 96.8%"
```

**In this project specifically**, that's not something that happens live, one flow at a time, while you watch a dashboard. The pipeline (`ids_ml/report.py::build_report` / `build_session_report`) runs the *entire* process, generate data, train, score every flow, generate alerts, **once, in a batch, ahead of time**, and writes every result into one `report.json` file plus an `alerts.db` database. The static website you browse (`ui/`) never runs a neural network at all; it just reads that precomputed file and renders tables and charts from it. This matches the project's actual goal: a deployable, static site with no live model server behind it, see [`docs/PRODUCT_VISION.md`](PRODUCT_VISION.md).

So "using the model" in this project looks like: *run a Python script once → get a report.json full of predictions and scores → the website just displays that file.* There's no moment where the website itself is "thinking."

---

## 5. Reality check: how big is this, really?

| | This project's MLP | This project's GRU (v3/v4) |
|---|---|---|
| Adjustable numbers (parameters) | 1,398 | 5,670 |
| Saved file size | a few KB | ~25 KB |
| Training examples seen | tens of thousands of synthetic flow rows | tens of thousands of flow rows, grouped into hundreds of sessions |
| Training time | under a minute | a few minutes |
| Hardware used | one CPU core in a WSL2 Linux environment on a regular laptop | same |
| What it can do | sort one flow into 1 of 6 categories | sort one whole session into 1 of 6 categories, using the order of flows |
| What it *can't* do | anything outside those 6 labels; it has no language understanding, no memory beyond one row, no reasoning | same, plus: only "remembers" the current session, nothing before or after it |

This is a genuinely tiny model by any modern standard, intentionally so. The project's whole point (see [`docs/PRODUCT_VISION.md`](PRODUCT_VISION.md)) is to demonstrate that even a small, cheap, fast-to-train network can out-perform hand-written rules on a task like this, not to build something enormous.

---

## 6. How does this compare to Claude, ChatGPT, and other flagship LLMs?

Here's the important part: **everything in Parts 2 and 3 above, weighted sums, activation functions, loss, backpropagation, Adam, epochs, early stopping, is exactly, literally what's happening inside Claude and ChatGPT too.** There is no secret extra ingredient. It's the same recipe. What's different is the scale, the architecture's shape, what it's trained to predict, and how many stages of training it goes through.

| | This project | Flagship LLMs (Claude, ChatGPT, etc.) |
|---|---|---|
| **Adjustable numbers (parameters)** | 1,398 – 5,670 | hundreds of **billions** (a number with 11+ digits) |
| **Training data** | ~tens of thousands of synthetic rows, generated by this project's own code | **trillions** of words/tokens scraped from books, websites, code, conversations, essentially a large fraction of all public text |
| **What one "input" looks like** | one network flow, or one session of flows | one chunk of text, broken into "tokens" (roughly word-pieces) |
| **What it's trained to predict** | pick 1 of 6 fixed labels (closed-set classification) | predict the *next token* in a sequence of text, one token at a time, over and over (open-ended generation) |
| **Core building block** | a small MLP, or a small GRU (reads a sequence by keeping one running summary as it goes, like reading left to right with a short-term memory) | a **Transformer**, reads an entire chunk of text at once, where every token can directly look at every other token (see "self-attention" below), not just a running summary |
| **Number of layers** | 2–3 layers | often 50-100+ stacked layers |
| **Training hardware** | one CPU core, a few minutes | thousands of specialized GPU chips, running for weeks to months |
| **Training stages** | one: supervised training on labeled examples, then done | several: (1) massive "pretraining" on next-token prediction across huge amounts of text, (2) "instruction tuning" on examples of good Q&A-style behavior, (3) **RLHF** (Reinforcement Learning from Human Feedback), human graders rank multiple model answers, and that ranking is used to further adjust the weights toward answers people actually prefer |
| **What "using" it looks like** | one fast, frozen forward pass per flow, batched ahead of time into a static report | one forward pass *per output token*, repeated live, re-feeding its own previous output back in as new input each time (this is why longer answers take proportionally longer to generate) |

### The one real architectural difference worth understanding: self-attention

This project's sequence model (the GRU) reads a session's flows **in order**, keeping one running "summary" number that gets updated as it goes, like reading a book one sentence at a time and updating your mental summary, but only ever remembering the gist, not the exact earlier sentences.

A Transformer (what Claude/ChatGPT are built from) instead lets **every token directly compare itself to every other token in the whole input, all at once**, weighted by how relevant they are to each other. It's less like reading in order and more like a meeting where everyone can simultaneously, directly reference anything anyone else in the room said, rather than relying on a summary passed down the line. This is *why* Transformers scaled so much better than older sequence models like GRUs/LSTMs for language, direct, full-context lookups beat passing a single running summary along, once you have enough compute to afford it. It's a different wiring diagram for the "stack of adjustable numbers," not a different kind of math.

### The throughline

Zoom out far enough and both this project's 1,398-number flow classifier and a 400-billion-parameter chat model are instances of the exact same idea: **take a pile of adjustable numbers, define a way to measure how wrong a guess is, and use calculus (backpropagation) plus a nudging rule (an optimizer like Adam) to make that pile of numbers slightly less wrong, millions of times in a row.** Everything else, Transformers vs. GRUs, classification vs. text generation, a laptop vs. a GPU datacenter, one training stage vs. four, is a difference of *scale and task*, not a difference of *fundamental mechanism*. Understanding this project's 1,398-number model honestly is understanding the real core of how every neural network, including the largest ones in the world, actually learns.
