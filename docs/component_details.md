Module Organization and Data Flow
┌─────────────┐
│ Audio Input │
└──────┬──────┘
       │
       ├────────────────────┬───────────────────┐
       ▼                    ▼                   │
┌──────────────┐     ┌──────────────┐           │
│Speech-to-Text│     │Tonal Analysis│           │
└──────┬───────┘     └──────┬───────┘           │
       │                    │                   │
       ▼                    │                   │
┌──────────────┐            │                   │
│Text Sentiment│            │                   │
│   Analysis   │            │                   │
└──────┬───────┘            │                   │
       │                    │                   │
       └────────┬───────────┘                   │
                ▼                               │     ┌─────────────────┐
       ┌────────────────┐                       │     │ User Preferences|
       │ LLM Processing │                       │     └─────────┬───────┘
       │ (Improvements) │───────────────────────────────────────│
       └────────┬───────┘                       │
                │                               │
                ▼                               │
       ┌────────────────┐                       │
       │Text-to-Speech  │◄──────────────────────┘
       │ (with cloned   │    (Original voice)
       │    voice)      │
       └────────────────┘


1. Whats the input data? - what are the features, what are the output labels?
2. Preprocessing steps? - vectorization, removing punctuation etc
3. Whats the train/test/val split?
4. Define the model - pretrained or from scratch
- If from scratch, what's the learning rate, how many neural network layers, whats the optimizer
- If from pretrained - what data the original model was trained on? what are the fine tuning steps?
5. Model.fit(train, val)
6. Model.evaluate(test)
7. Graphs/analysis on performance metrics like accuracy, F1, Loss graph, learning curves


Unit tests:

assert input = expected input
assert output = expected output
