import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
plt.axis("off")

sys.path.insert(1, 'Visual/Models/Training')
from BaseModel import *

model = tf.keras.models.load_model("Numerical/Models/Saved/Hilbert.h5")

labels = np.array(sorted(os.listdir("Numerical/Datasets/HilbertCurves/Hilbert")))
print(labels)

k = 4
shape = (64, 64, 3)

def read_image(file_name):
  image = tf.keras.utils.load_img(file_name, target_size=(shape[0], shape[1]))
  input_arr = tf.keras.utils.img_to_array(image)
  image = np.array(image)
  return image

def interpolate_images(baseline,
                       image,
                       alphas):
  alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  input_x = tf.expand_dims(image, axis=0)
  delta = tf.cast(input_x, float) - tf.cast(baseline_x, float)
  images = baseline_x +  alphas_x * delta
  return images

baseline = tf.zeros(shape=shape)

def top_k_predictions(img, k=k):
  image_batch = tf.expand_dims(img, 0)
  predictions = model(image_batch)
  probs = tf.nn.softmax(predictions, axis=-1)
  top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
  print(top_probs, top_idxs)
  top_labels = labels[tuple(top_idxs)]
  return top_labels, top_probs[0]

def compute_gradients(images, target_class_idx):
  with tf.GradientTape() as tape:
    tape.watch(images)
    logits = model(images)
    probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
  return tape.gradient(probs, images)

def integral_approximation(gradients):
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients

def integrated_gradients(baseline,
                         image,
                         target_class_idx,
                         m_steps=50,
                         batch_size=32):
  alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
  gradient_batches = []

  for alpha in tf.range(0, len(alphas), batch_size):
    from_ = alpha
    to = tf.minimum(from_ + batch_size, len(alphas))
    alpha_batch = alphas[from_:to]

    gradient_batch = one_batch(baseline, image, alpha_batch, target_class_idx)
    gradient_batches.append(gradient_batch)
  total_gradients = tf.concat(gradient_batches, axis=0)

  avg_gradients = integral_approximation(gradients=total_gradients)

  integrated_gradients = (image - baseline) * avg_gradients

  return integrated_gradients

@tf.function
def one_batch(baseline, image, alpha_batch, target_class_idx):
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch)

    gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                       target_class_idx=target_class_idx)
    return gradient_batch

def plot_img_attributions(baseline,
                          image,
                          target_class_idx,
                          save_path,
                          m_steps=50,
                          cmap=None,
                          overlay_alpha=0.4):

  attributions = integrated_gradients(baseline=baseline,
                                      image=image,
                                      target_class_idx=target_class_idx,
                                      m_steps=m_steps)
  attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

  plt.imshow(attribution_mask, cmap=cmap)
  plt.tight_layout()
  plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)

path_to_saliency = "Numerical/Models/Saliency/Input/"
save_path = "Numerical/Models/Saliency/Saved/"

for file in os.listdir(path_to_saliency):
  image = read_image(f"{path_to_saliency}{file}")
  imglabels, probs = top_k_predictions(image, k=k)
  idx = list(labels).index(imglabels[0])
  plot_img_attributions(image=image,
                            baseline=baseline,
                            target_class_idx=idx,
                            save_path=f"{save_path}{file}",
                            m_steps=240,
                            cmap=plt.cm.inferno,
                            overlay_alpha=0.4)
