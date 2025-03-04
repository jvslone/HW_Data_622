# **Assignment: Implementing and Extending a Neural Network for Wine Quality Classification**

## **Objective**
In this assignment, you will:
1. **Implement a neural network from scratch** using only NumPy (without PyTorch, TensorFlow, or any deep learning libraries).
2. Train the network on the **UCI Wine Quality dataset** to classify wines based on their chemical attributes.
3. Extend your understanding by explaining how a trained neural network can be **used for data generation**, paving the way for generative AI models.

---

## **Part 1: Implement a Neural Network from Scratch**

### **Task Overview**
You are provided with a **template Python script** where the main functions (`forward_propagation`, `backward_propagation`, activation functions, and loss computation) are missing. Your task is to **fill in the missing implementations** based on the comments and instructions provided in the template.

### **Steps to Complete**
1. **Download the UCI Wine Quality dataset** (red wine) using the provided URL.
2. **Preprocess the dataset**:
   - Normalize the features.
   - Convert wine quality scores into a classification task.
   - Split the dataset into training and testing sets.
3. **Implement the missing functions**:
   - **ReLU Activation Function** (`relu`): Ensures only positive values pass through.
   - **Softmax Activation Function** (`softmax`): Converts raw scores into probabilities.
   - **Forward Propagation** (`forward_propagation`): Computes predictions given input.
   - **Loss Computation** (`compute_loss`): Uses **categorical cross-entropy**.
   - **Backward Propagation** (`backward_propagation`): Updates weights via **gradient descent**.
4. **Train your network**:
   - Run for **1,000 epochs** and observe loss decreasing.
5. **Evaluate the model**:
   - Compute accuracy on the test set.

### **Deliverables**
- Your completed Python script (`wine_nn.py` or `.ipynb` file).
- A short (1-2 page) report including:
  - A SHORT description of how your neural network **learns**.
  - Graphs/plots of loss over time.
  - Test set accuracy.

---

## **Part 2: Explain How a Neural Network Can Generate New Data**

### **Task Overview**
Once a neural network is trained, it can be **used beyond just classification**. Your task in this section is to **explain how a neural network can generate new data**.

### **Topics to Cover in Your Explanation**
In a **1-2 page response**, address the following:
1. **How can a standard feedforward neural network be used to generate synthetic data?**  
   - Discuss how you might **sample inputs** and use the trained model to create new instances that resemble real data.
   
2. **What modifications are needed to move from classification to generation?**  
   - Can a network trained for classification be **repurposed** for generating new data?
   - What changes (e.g., adding noise, modifying outputs) might help?

3. **What types of generative models extend this idea?**  
   - Briefly mention **autoencoders** (compressing & reconstructing data).
   - Discuss how **GANs (Generative Adversarial Networks)** use a discriminator to refine generated data.
   - If applicable, include an example of how you could modify your wine classifier into a generative model.

### **Deliverables**
- A 1-2 page write-up titled **"Using Neural Networks for Data Generation"** (PDF or Word).

---

## **Grading Criteria (100 Points)**

| Criteria                                                  | Points |
|-----------------------------------------------------------|--------|
| **Part 1: Implementation of Neural Network**              | 50    |
| - ReLU & Softmax functions implemented correctly          | 10    |
| - Forward propagation computes correct activations        | 10    |
| - Loss function correctly implemented                     | 10    |
| - Backpropagation updates weights properly                | 10    |
| - Model trains and achieves reasonable accuracy           | 10    |
| **Part 2: Theoretical Explanation**                       | 30    |
| - Explanation of how a trained NN can generate new data   | 10    |
| - Discussion of modifications needed for generative tasks | 10    |
| - Overview of autoencoders/GANs for generative modeling   | 10    |
| **Report Clarity & Completeness**                         | 20    |
| - Code is well-documented and structured                  | 10    |
| - Write-up is clear, well-reasoned, and complete          | 10    |

---

## **Submission Instructions**
- Submit the link to a repository containing all of your files:
  1. Your **completed Python script** (`nn.py` or `.ipynb`).
  2. A **short report** on your implementation and model accuracy.
  3. Your **1-2 page explanation** on how NNs can generate data.

- **Deadline:** 📅 Feb 27th 11:59pm 

---

## **Hints & Tips**
- **Debugging Tip:** Print the shape of matrices at each step to ensure consistency.
- **Performance Tip:** Be wary of using your own computer for this work.
- **Commit your code OFTEN!**
- **10 point Extra Credit Opportunity:** Modify your model to generate **synthetic wine quality data** and include sample outputs in your report. 🚀

---

