from sklearn.metrics import log_loss
import numpy as np

y_true = [0,0,1,1]
y_pred = [[0.1,0.9], [0.2,0.8], [0.3,0.7], [0.01, 0.99]]
sk_log_loss = log_loss(y_true,y_pred)
print('Loss by sklearn: %s.'%sk_log_loss)