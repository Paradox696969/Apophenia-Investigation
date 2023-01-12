import random
import string
import pandas as pd
import os

stochasticity = {"None": (0, 0), "Low-Stochas": (0, 6), "High-Stochas": (0, 2), "Pure-Stochas": (1, 1)}
dataDict = {"String":[], "LastLetter":[]}

str1 = ""
# save_path = "Textual/Datasets/training.csv"
required_samples = 100000
max_num_per_sample = 4
printable = string.ascii_letters + string.digits

stochasticities = list(stochasticity.keys())[:-3]
for x in range(required_samples):
    stochasLevel = random.choice(stochasticities)
    for i in range(random.randint(1, 10)):
        str1 += random.choice(string.ascii_letters + string.digits)
    
    str1 += ", "
    str1 = (str1*max_num_per_sample)[:-2]
    
    strlist = list(str1)
    for stritem in range(len(strlist)):
        item = strlist[stritem]
        if item not in [",", " "]:
            choice = random.choice(printable)
            strlist[stritem] = choice if random.randint(*stochasticity[stochasLevel]) == 1 and choice not in ["/", "."]  else item
    
    dataDict["LastLetter"].append(strlist.pop(-1))
    dataDict["String"].append("".join(strlist))
    print("".join(strlist))
    str1 = ""

# df = pd.DataFrame(dataDict)
# df = df.reset_index()
# del df['index']
# df.to_csv(save_path, index=False)
for i in range(len(dataDict["LastLetter"])):
    try:
        os.mkdir(f"Textual/Datasets/Text/{dataDict['LastLetter'][i]}/")
    except:
        ...
    

    f = open(f"Textual/Datasets/Text/{dataDict['LastLetter'][i]}/{i}.txt", "w")
    f.write(dataDict["String"][i])
    f.close()

