from template import train_painting_template
from template_2 import imagenet_templates
import numpy as np
import pandas as pd


# concepts = ["Van Gogh","Claude Monet","Pablo Picasso"]

# csv_rows = []
# for concept in concepts:
#     for template in train_painting_template:
#         prompt = template.format(concept)
#         seed = np.random.randint(0, 10000)
#         csv_rows.append({"prompt": prompt, "sd_seed": seed})

# df = pd.DataFrame(csv_rows)
# df["idx"] = range(len(df))
# df = df[["idx", "prompt", "sd_seed"]]
# df.to_csv("style_3.csv", index=False)

csv_rows = []
for template in imagenet_templates:
    prompt = template.format("snoopy")
    seed = np.random.randint(0, 10000)
    csv_rows.append({"prompt": prompt, "sd_seed": seed})

for template in train_painting_template:
    prompt = template.format("Van Gogh")
    seed = np.random.randint(0, 10000)
    csv_rows.append({"prompt": prompt, "sd_seed": seed})

df = pd.DataFrame(csv_rows)
df["idx"] = range(len(df))
df = df[["idx", "prompt", "sd_seed"]]
df.to_csv("van_snoopy.csv", index=False)