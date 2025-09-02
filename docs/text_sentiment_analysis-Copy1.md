## 1. Input Data

### a. Data  
The source files are loaded from 3 files in the MELD dataset:  
- `train_sent_emo.csv`  
- `dev_sent_emo.csv`  
- `test_sent_emo.csv`  

### b. Features  
Each dataset contains two columns:  
- **Utterance** (*stored as X*): quotes spoken by characters in the Friends TV show, transcribed into text for analysis.  
- **Emotion** (*stored as y*): the true emotional label for each quote, determined based on multiple factors, including—but not limited to—the semantics of the text.  

## 2. Preprocessing Steps

### a. Mapping  
The model was trained on a dataset containing seven emotions:  
- **0:** anger  
- **1:** disgust  
- **2:** fear  
- **3:** joy  
- **4:** neutral  
- **5:** sadness  
- **6:** surprise  

The *Emotion* column was converted into integer codes using the `label2id` dictionary.  
Any rows with emotions outside these categories were dropped from the dataset.  

---

### b. Dataset Distribution  
The dataset is divided into training, development (validation), and test sets (80-20 split):  
- **Training set:** 9,989 instances  
- **Development set (validation set):** 1,109 instances  
  - **Purpose of the development set:**  
    - It’s a **validation set**, separate from training and test data.  
    - Used during training to:  
      1. **Monitor performance** after each epoch (e.g., validation accuracy/loss).  
      2. **Tune hyperparameters** such as learning rate, batch size, and number of layers.  
      3. **Apply early stopping** — training stops when validation accuracy stops improving, to avoid overfitting.  
      4. **Guide model selection** — compare multiple architectures using validation performance.  
- **Test set:** 2,610 instances  

---

### c. Label Distribution (Training Set)  
The label counts within the training set are as follows:  
- Neutral: 4,710  
- Joy: 1,743  
- Surprise: 1,205  
- Anger: 1,109  
- Sadness: 683  
- Disgust: 271  
- Fear: 268  

---

### d. Cleaning  
The dataset was processed so that:  
- All text was lowercased.  
- Any characters that were not whitespace, letters, digits, or basic punctuation were replaced with a space.  

---

### e. Tokenizer  
- A tokenizer was applied to map words in *Utterance* to integers.  
- The tokenizer also built a vocabulary by learning the frequency of each word’s occurrence, e.g.:  
  ```json
  {"the": 1, "i": 2, "you": 3, ...}

## 3. Define the Model — Pretrained or From Scratch

### a. Labels  
The labels from the training and test sets were converted into integer form, so that all true emotions across the three datasets are represented as integers.  

---

### b. Model Architecture  
The model consists of three main components:  

1. **Embedding Layer**  
   - Converts word indices into dense vectors.  
   - Example: `word 57 → [0.12, -0.09, ..., 0.34]`.  
   - Embedding dimension = **128**, providing multiple features per word.  
   - Helps the neural network learn semantic relationships between words and capture more nuance.  

2. **Bidirectional LSTM Layers (BiLSTM)**  
   - Based on **Long Short-Term Memory (LSTM)**, a type of RNN that remembers past context with its internal memory.  
   - Processes sequences in **both directions**:  
     - Forward: *I → am → happy*  
     - Backward: *happy → am → I*  
   - This bidirectional context improves interpretation of emotions.  

3. **Dense (Fully Connected) Layers**  
   - Takes the LSTM outputs and passes them through dense layers.  
   - Final layer uses **softmax** to output probabilities across **7 emotion classes**:  
     ```
     [0.01 anger, 0.02 disgust, 0.05 fear, 0.70 joy, 
      0.10 neutral, 0.07 sadness, 0.05 surprise]
     ```  
   - The highest probability corresponds to the predicted emotion.  
   
## 4. Building the Model

### a. Vocab Size  
Determines the size of the embedding matrix by calculating the number of unique words in the training set.  

### b. Embedding Dimension  
Set to **128 features/dimensions** — the size of each word vector. This defines how many features represent each word in the embedding space.  

### c. Max Sequence Length (`max_len`)  
Defines the maximum number of words/tokens allowed per sentence when preparing the data. Sequences longer than this are truncated, and shorter ones are padded.  

### d. Number of Classes (`num_classes`)  
Specifies the number of output labels. In this case: **7 emotions** (anger, disgust, fear, joy, neutral, sadness, surprise).  

## 5. Compile the Model

### a. Optimizer (Adam)  
The **Adam optimizer** adjusts model weights during backpropagation.  
- **Forward propagation:** the model predicts probabilities across the 7 emotions.  
- **Loss calculation:** the predicted probabilities are compared with the true labels.  
- **Backpropagation:** the optimizer updates the weights to reduce this loss.  

Adam adapts the **learning rate** for each parameter based on past gradients, making training more efficient and stable.  

**Learning rate behavior:**  
- Too small → training is slow; risk of getting stuck in a local minimum.  
- Too large → optimizer may overshoot the minimum.  
- Adam helps stabilize training by steadily decreasing the loss toward a local minimum.  

---

### b. Loss Function (`sparse_categorical_crossentropy`)  
- The model outputs a probability distribution across the 7 emotion classes.  
- The loss function compares this predicted distribution with the true label (stored as an integer, e.g., `3` for *joy*).  
- Internally, this is treated as if compared to a one-hot vector: Joy = [0, 0, 0, 1, 0, 0, 0]

**Interpretation:**  
- **High loss** → model assigned low probability to the correct class.  
- **Low loss** → model was closer to the right answer.  
- The loss function guides weight updates during training and measures how well the model is learning.  

---

### c. Metrics  
The model reports **accuracy** during training and evaluation.  
- Accuracy = proportion of predictions that match the true labels.  

## 6. Model Summary  

The **model summary** provides a structured overview of the model:  
- **Architecture**: lists each layer in the model in the order they are applied.  
- **Output shapes**: shows the dimensions of the data as it flows through each layer.  
- **Parameters**: displays the number of trainable weights and biases for each layer.  

This summary helps verify that the model is built correctly, ensures parameter counts match expectations, and gives insight into the overall complexity of the network.  

## 7. Early Stopping  

**Purpose:** Prevents **overfitting** and saves time by stopping training once the model stops improving.  

### Key Parameters:  
- **`monitor='val_accuracy'`** → watches validation accuracy after each epoch.  
- **`patience=5`** → if validation accuracy doesn’t improve for 5 consecutive epochs, stop training early.  
- **`restore_best_weights=True`** → after stopping, roll back to the weights from the epoch with the **best validation accuracy** (instead of keeping the last, worse weights).  
- **`verbose=1`** → prints a message when early stopping is triggered.  

## 8. ReduceLROnPlateau  

**Purpose:** Helps the optimizer take **smaller, finer steps** when progress slows, instead of bouncing around the minimum.  

### Key Parameters:  
- **`monitor='val_loss'`** → watches validation loss.  
- **`patience=3`** → if validation loss doesn’t improve for 3 epochs, reduce the learning rate.  
- **`factor=0.5`** → when triggered, multiply the current learning rate by 0.5 (cut it in half).  
- **`min_lr=1e-7`** → prevents reducing the learning rate below this minimum.  
- **`verbose=1`** → prints a message when the learning rate is reduced.  

---

### Example Timeline  
Suppose the learning rate starts at `0.001`:  

![Learning Rate Example](f94104d5-2db8-450c-90c1-abf4e2fc250e.png)  

In this example:  
- Epoch 1–2 → validation loss improves → keep LR at `0.001`.  
- Epoch 3–4 → no improvement → wait (1/3, 2/3).  
- Epoch 5 → still no improvement → **trigger** → new LR = `0.0005`.  

## 9. Training  

### Key Settings:  
- **`X_train_pad, y_train`** → training inputs (padded sequences) and labels (emotion classes).  
- **`batch_size=32`** → process 32 samples at a time before updating weights.  
- **`epochs=20`** → train for up to 20 passes through the dataset (may stop earlier if EarlyStopping is triggered).  
- **`validation_data=(X_dev_pad, y_dev)`** → after each epoch, evaluate on the validation set to monitor progress.  
- **`callbacks=callbacks`** → includes `EarlyStopping` and `ReduceLROnPlateau`.  
- **`verbose=1`** → prints progress for each epoch (loss, accuracy, val_loss, val_accuracy).  

### Notes:  
- The returned `history` object stores all metrics per epoch.  
- These metrics can be used later for plotting **training vs. validation accuracy/loss curves**.  

## 10. Evaluate on Test Set  

- **`model.evaluate(...)`** → runs the trained model on the held-out test set.  
- **Computes final `loss` and `accuracy`** → measures model performance on unseen data.  
- **`verbose=0`** → suppresses progress bar, only returns results.  

## 11. Graphs / Analysis on Performance  

### Training vs Validation Metrics  
- **Left graph:** Model Accuracy over epochs (Training vs Validation).  
- **Right graph:** Model Loss over epochs (Training vs Validation).  

![Model Accuracy and Loss](file-128EmsnEqK56ry1QSSYCJo)

---

### Classification Report  
The classification performance across the 7 emotions:  

![Classification Report](file-128EmsnEqK56ry1QSSYCJo)

---

📊 Observations:  
- Stronger performance on **neutral** and **surprise**.  
- Moderate results for **joy**.  
- Poor performance on low-support classes (**disgust** and **fear**) due to class imbalance.  
- **Overall Accuracy:** `0.5042`.  





 
