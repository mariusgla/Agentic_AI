out_dir = 'out-chess'
dataset = 'chess'
block_size = 128            # wie in prepare.py

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

# GTX 1070: 8 GB VRAM, kein bf16, kein compile
dtype = 'float32'
compile = False
batch_size = 64
gradient_accumulation_steps = 4     # effektive Batchgröße 256

learning_rate = 1e-3
min_lr = 1e-4
max_iters = 14000                   # ~5 Epochen
lr_decay_iters = 14000
warmup_iters = 200
beta2 = 0.99

eval_interval = 500
eval_iters = 50
log_interval = 10