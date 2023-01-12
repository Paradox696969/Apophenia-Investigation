import matplotlib
from matplotlib import pyplot as plt
# Preset Matplotlib figure sizes.
matplotlib.rcParams['figure.figsize'] = [9, 6]

import tensorflow as tf
print(tf.__version__)
# set random seed for reproducible results 
tf.random.set_seed(22)
x_vals = tf.linspace(-5, 5, 201)
x_vals = tf.cast(x_vals, tf.float32)

def loss(x):
  # return 2*(x**4) + 3*(x**3) + 2*(x**2) + 2
  return tf.abs(tf.math.sin(x))+1

def grad(f, x):
  with tf.GradientTape() as tape:
    tape.watch(x)
    result = f(x)
  return tape.gradient(result, x)

def sigmoid_base(x):
  return 1 / (1 + tf.math.exp(x))

def sigmoid_var(x, itertime, flatten=1):
  return sigmoid_base(flatten*x-itertime)+0.01

class SigNadam(tf.Module):

    def __init__(self, learning_rate=1e-3, beta_1=0.999, beta_2=0.999, ep=1e-7):
      # Initialize the Adam parameters
      self.beta_1 = beta_1
      self.beta_2 = beta_2
      self.learning_rate = learning_rate
      self.ep = ep
      self.t = 1
      self.v_dvar, self.s_dvar = [], []
      self.title = f"SigNadam: learning rate={self.learning_rate}"
      self.built = False

    def apply_gradients(self, grads, vars, itertime, flatten):
      # Set up moment and RMSprop slots for each variable on the first call
      if not self.built:
        for var in vars:
          v = tf.Variable(tf.zeros(shape=var.shape))
          s = tf.Variable(tf.zeros(shape=var.shape))
          self.v_dvar.append(v)
          self.s_dvar.append(s)
        self.built = True
      # Perform Adam updates
      for i, (d_var, var) in enumerate(zip(grads, vars)):
        # Nadam
        self.v_dvar[i] = self.beta_1*self.v_dvar[i] + (1-self.beta_1)*d_var
        self.s_dvar[i] = self.beta_2*self.s_dvar[i] + (1-self.beta_2)*tf.square(d_var)
        v_dvar_bc = self.v_dvar[i]/(1-(self.beta_1**self.t))
        s_dvar_bc = self.s_dvar[i]/(1-(self.beta_2**self.t))
        v_dvar_bc_vinc = self.beta_1*v_dvar_bc + (1-self.beta_1)*d_var

        #SigNadam
        beta_sig = self.beta_1 * sigmoid_var(float(self.t), float(itertime), flatten)
        v_dvar_bc_vinc = beta_sig*v_dvar_bc + (1-self.beta_1)*d_var

        # Update model variables
        var.assign_sub(self.learning_rate*((v_dvar_bc_vinc)/(tf.sqrt(s_dvar_bc) + self.ep)))
      # Increment the iteration counter
      self.t += 1.

def convergence_test(optimizer, loss_fn, grad_fn=grad, init_val=2., max_iters=5000):
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
    optimizer.apply_gradients([x_grad], [x_star], 1000, 1)
    param_path.append(x_star.numpy())

    # Checking for convergence
    if round(float(x_star), 5) == round(float(x_old), 5):
      print(f"Converged in {iter} iterations\n")
      converged = True
      break
    # if (x_star) == float(x_old):
    #   print(f"Converged in {iter} iterations\n")
    #   converged = True
    #   break

  # Print early termination message
  if not converged:
    print(f"Exceeded maximum of {max_iters} iterations. Test terminated.\n")
  return param_path

param_map_gd = {}
learning_rates = [1e-3, 1e-2, 1e-1]
for learning_rate in learning_rates:
  param_map_gd[learning_rate] = (convergence_test(
      SigNadam(learning_rate=learning_rate), loss_fn=loss))

def viz_paths(param_map, x_vals, loss_fn, title, max_iters=5000):
  # Creating a controur plot of the loss function
  t_vals = tf.range(1., max_iters + 100.)
  t_grid, x_grid = tf.meshgrid(t_vals, x_vals)
  loss_grid = tf.math.log(loss_fn(x_grid))
  plt.pcolormesh(t_vals, x_vals, loss_grid, shading='nearest')
  colors = ['r', 'b', 'c']
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