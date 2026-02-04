# Activations-Geometry

**Swish Activation Function:**  

$$f(x) = x \cdot \frac{1}{1 + e^{-x}}$$

**Mish Activation Function:**  
$$f(x) = x \cdot \tanh(\ln(1 + e^x))$$

Unlike the ReLU activation function, learning still happens even when the input is negative.  
The gradients of both Swish and Mish are smooth, which helps optimization.
