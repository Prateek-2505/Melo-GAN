import json, numpy as np, matplotlib.pyplot as plt, os

# path to a saved loss file (update if different)
loss_file = "data/experiments/gan/ae_loss_history.json"
if os.path.exists(loss_file):
    with open(loss_file) as f: hist = json.load(f)
    train = hist['train_loss']; val = hist.get('val_loss', None)
else:
    # fallback: try npy
    if os.path.exists("data/experiments/gan/ae_loss_history.npy"):
        hist = np.load("data/experiments/gan/ae_loss_history.npy", allow_pickle=True).item()
        train = hist['train_loss']; val = hist.get('val_loss', None)
    else:
        raise SystemExit("No AE loss history found at data/experiments/gan/")

plt.plot(train, label='train loss')
if val is not None: plt.plot(val, label='val loss')
plt.yscale('log')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()
plt.title('AE reconstruction loss')
plt.show()
