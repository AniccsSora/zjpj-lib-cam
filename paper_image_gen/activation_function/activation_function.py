import numpy as np
import matplotlib.pyplot as plt


# Redefine activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softsign(x):
    return x / (1 + np.abs(x))

def softsign_derivative(x):
    return 1 / (1 + np.abs(x))**2

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    sig_x = sigmoid(x)
    return sig_x + x * sig_x * (1 - sig_x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha):
    return np.where(x > 0, x, x * alpha)

def leaky_relu_derivative(x, alpha):
    return np.where(x > 0, 1, alpha)
def draw_activation_functions():
    # Superparameter for font sizes
    font_size = 14
    title_size = font_size + 2

    rows, cols = 3, 2

    x = np.linspace(-5, 5, 100)
    sp_x = np.linspace(-10, 10, 400)
    plt.figure(figsize=(8, 10))

    # change default font size
    plt.rcParams.update({'font.size': font_size})

    # Sigmoid
    plt.subplot(rows, cols, 1)
    plt.plot(x, sigmoid(x), label='Sigmoid')
    plt.title('Sigmoid', fontsize=title_size)
    plt.grid(True)

    # TanH
    plt.subplot(rows, cols, 2)
    plt.plot(x, tanh(x), label='TanH')
    plt.title('TanH', fontsize=title_size)
    plt.grid(True)

    # Softsign
    plt.subplot(rows, cols, 3)
    plt.plot(sp_x, softsign(sp_x), label='Softsign')
    plt.title('Softsign', fontsize=title_size)
    plt.grid(True)

    # Swish
    plt.subplot(rows, cols, 4)
    plt.plot(x, swish(x), label='Swish')
    plt.title('Swish', fontsize=title_size)
    plt.grid(True)

    # relu
    plt.subplot(rows, cols, 5)
    plt.plot(x, relu(x), label='relu')
    plt.title('ReLU', fontsize=title_size)
    plt.grid(True)

    # leaky relu
    plt.subplot(rows, cols, 6)
    plt.plot(x, leaky_relu(x, 0.1), label='leakage \nfactor = 0.1')

    # desc message
    plt.text(-5, 0.8, r'$\alpha=0.1$', fontsize=font_size)
    plt.title('Leaky ReLU', fontsize=title_size)
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Optionally adjust global font size for labels, ticks, etc.
    plt.rcParams.update({'font.size': font_size})

    # Save the figure
    plt.savefig('activation_functions_with_superparameter.png')
    # show image
    plt.show()

    print("The activation functions plot has been saved as 'activation_functions_with_superparameter.png'.")


# Drawing function for activation functions and their derivatives
def draw_derivatives_act_functions():
    font_size = 14
    title_size = font_size + 2

    # use next color index
    plt.rcParams.update({'axes.prop_cycle': plt.cycler(color=plt.cm.Set2.colors)})

    x = np.linspace(-5, 5, 200)
    sp_x = np.linspace(-10, 10, 400)
    plt.figure(figsize=(8, 10))  # Adjust size as needed

    plt.rcParams.update({'font.size': font_size})

    # Sigmoid Derivative
    plt.subplot(3, 2, 1)
    plt.plot(x, sigmoid_derivative(x), label='Sigmoid Derivative')
    plt.title('Sigmoid Derivative', fontsize=title_size)
    plt.grid(True)

    # TanH Derivative
    plt.subplot(3, 2, 2)
    plt.plot(x, tanh_derivative(x), label='TanH Derivative')
    plt.title('TanH Derivative', fontsize=title_size)
    plt.grid(True)

    # Softsign Derivative
    plt.subplot(3, 2, 3)
    plt.plot(sp_x, softsign_derivative(sp_x), label='Softsign Derivative')
    plt.title('Softsign Derivative', fontsize=title_size)
    plt.grid(True)

    # Swish Derivative
    plt.subplot(3, 2, 4)
    plt.plot(x, swish_derivative(x), label='Swish Derivative')
    plt.title('Swish Derivative', fontsize=title_size)
    plt.grid(True)

    # ReLU Derivative
    plt.subplot(3, 2, 5)
    plt.plot(x, relu_derivative(x), label='ReLU Derivative')
    plt.title('ReLU Derivative', fontsize=title_size)
    plt.grid(True)

    # Leaky ReLU Derivative
    plt.subplot(3, 2, 6)
    plt.plot(x, leaky_relu_derivative(x, alpha=0.1), label='Leaky ReLU Derivative')
    # force draw tick of y-axis
    plt.ylim(-0.1, 1.1)
    # draw specified tick on y-axis
    plt.yticks([0, 1, 0.1, 0.25, 0.5,0.75])
    plt.title('Leaky ReLU Derivative', fontsize=title_size)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('activation_functions_derivatives.png')
    plt.show()
if __name__ == '__main__':
    draw_activation_functions()

    draw_derivatives_act_functions()

