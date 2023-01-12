from BaseModel import *

path = "Testing/"
result_str = ""

model = tf.keras.models.load_model("Visual/Models/Saved/MergedImprovedEvenModel.h5")

pred_ds = loadPredDs(path, 2, 120, 120)
result = predEval(pred_ds, model)
result_str += f"\n{result[1]['accuracy']}"

print(result_str)
