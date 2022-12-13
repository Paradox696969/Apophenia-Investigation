import matplotlib
from matplotlib import pyplot as plt
# Preset Matplotlib figure sizes.
matplotlib.rcParams['figure.figsize'] = [9, 6]

import tensorflow as tf
print(tf.__version__)
# set random seed for reproducible results 
tf.random.set_seed(22)

class Momentum(tf.Module):

  def __init__(self, learning_rate=1e-3, momentum=0.7):
    # Initialize parameters
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.change = 0.
    self.title = f"Gradient descent optimizer: learning rate={self.learning_rate}"

  def apply_gradients(self, grads, vars):
    # Update variables 
    for grad, var in zip(grads, vars):
      curr_change = self.learning_rate*grad + self.momentum*self.change
      var.assign_sub(curr_change)
      self.change = curr_change

x_vals = tf.linspace(-2, 2, 201)
x_vals = tf.cast(x_vals, tf.float32)

def loss(x):
  return 2*(x**4) + 3*(x**3) + 2

def grad(f, x):
  with tf.GradientTape() as tape:
    tape.watch(x)
    result = f(x)
  return tape.gradient(result, x)

def convergence_test(optimizer, loss_fn, grad_fn=grad, init_val=2., max_iters=2000):
  # Function for optimizer convergence test
  print(optimizer.title)
  print("-------------------------------")
  # Initializing variables and structures
  x_star = tf.Variable(init_val)
  param_path = []
  converged = False

  for iter in range(1, max_iters + 1):
    x_grad = grad_fn(loss_fn, x_star)

    # Case for exploding gradient
    if tf.math.is_nan(x_grad):
      print(f"Gradient exploded at iteration {iter}\n")
      return []

    # Updating the variable and storing its old-version
    x_old = x_star.numpy()
    optimizer.apply_gradients([x_grad], [x_star])
    param_path.append(x_star.numpy())

    # Checking for convergence
    if x_star == x_old:
      print(f"Converged in {iter} iterations\n")
      converged = True
      break

  # Print early termination message
  if not converged:
    print(f"Exceeded maximum of {max_iters} iterations. Test terminated.\n")
  return param_path

param_map_gd = {}
learning_rates = [1e-3, 1e-2, 1e-1]
for learning_rate in learning_rates:
  param_map_gd[learning_rate] = (convergence_test(
      Momentum(learning_rate=learning_rate), loss_fn=loss))

def viz_paths(param_map, x_vals, loss_fn, title, max_iters=2000):
  # Creating a controur plot of the loss function
  t_vals = tf.range(1., max_iters + 100.)
  t_grid, x_grid = tf.meshgrid(t_vals, x_vals)
  loss_grid = tf.math.log(loss_fn(x_grid))
  plt.pcolormesh(t_vals, x_vals, loss_grid, vmin=0, shading='nearest')
  colors = ['r', 'w', 'c']
  # Plotting the parameter paths over the contour plot
  for i, learning_rate in enumerate(param_map):
    param_path = param_map[learning_rate]
    if len(param_path) > 0:
      x_star = param_path[-1]
      plt.plot(t_vals[:len(param_path)], param_path, c=colors[i])
      plt.plot(len(param_path), x_star, marker='o', c=colors[i], 
              label = f"x*: learning rate={learning_rate}")
  plt.xlabel("Iterations")
  plt.ylabel("Parameter value")
  plt.legend()
  plt.title(f"{title} parameter paths")
  plt.show()

viz_paths(param_map_gd, x_vals, loss, "Gradient descent")