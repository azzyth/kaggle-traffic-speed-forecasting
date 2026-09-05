import numpy as np
import pandas as pd

data = np.load(r'D:\coding stuff\pandasenv\kaggle\train\test\test_X_hist.npy')

# Reshape: Menggabungkan (15, 1260) menjadi 18900
data_reshaped = data.reshape(data.shape[0], -1) 

df = pd.DataFrame(data_reshaped)
df.to_csv('test_flattened.csv', index=False)
print(f"Data berhasil diekspor dengan shape: {df.shape}")