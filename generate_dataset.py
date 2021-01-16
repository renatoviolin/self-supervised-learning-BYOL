# %%
import glob
import os
import uuid


# %%
cats = glob.glob('data/Cat/*.jpg')
dogs = glob.glob('data/Dog/*.jpg')
target_dir = 'data/no_label'

# %%
for c_path in cats[:124]:
    name = str(uuid.uuid4()).split('-')[0]
    target_path = f'{target_dir}/{name}.jpg'
    os.rename(c_path,target_path)
    

# %%
for c_path in dogs[:125]:
    name = str(uuid.uuid4()).split('-')[0]
    target_path = f'{target_dir}/{name}.jpg'
    os.rename(c_path,target_path)
    