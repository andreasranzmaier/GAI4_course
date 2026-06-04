"""Patch NVIDIA pix2pixHD for Python 3.11+ 

Run from Unit_5_pix2pixHD/ (the folder that contains pix2pixHD/):
    python apply_patches.py

"""
import os

BASE = "pix2pixHD"


def patch_file(path, replacements, guard=None):
    """Apply (old, new) string replacements to a file. Skip if `guard` already present."""
    with open(path, "r") as f:
        txt = f.read()
    if guard is not None and guard in txt:
        print(f"  skip  {path} (already patched)")
        return
    for old, new in replacements:
        txt = txt.replace(old, new)
    with open(path, "w") as f:
        f.write(txt)
    print(f"  patch {path}")


# 1. train.py
#    - swap `import fractions` for `import math` (fractions.gcd removed in Py 3.9)
#    - cast lcm args to int (print_freq * batchSize can be float, math.gcd needs int)
#    - define model_ref inside training loop so it works without DataParallel (CPU / single GPU)
#    - save checkpoints at milestone epochs 5, 10, 20, 40
train_replacements = [
    ("import fractions\n", "import math\n"),
    (
        "def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0",
        "def lcm(a,b): return abs(a * b)/math.gcd(int(a), int(b)) if a and b else 0",
    ),
    # Insert model_ref at the top of the per-iteration loop body, just before forward pass
    (
        "        # whether to collect output images\n        save_fake = total_steps % opt.display_freq == display_delta\n\n        ############## Forward Pass ######################",
        "        # whether to collect output images\n        save_fake = total_steps % opt.display_freq == display_delta\n\n        model_ref = model.module if hasattr(model, 'module') else model\n        ############## Forward Pass ######################",
    ),
    # Outer (script-init) optimizer assignment — model_ref isn't in scope here yet,
    # so use the conditional pattern. create_model() only wraps in DataParallel for
    # multi-GPU; on CPU / single-GPU it returns the bare model without `.module`.
    (
        "else:\n    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D",
        "else:\n    if hasattr(model, 'module'):\n        optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D\n    else:\n        optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D",
    ),
    # In-loop calls — model_ref is defined above, safe to use
    ("model.module.save", "model_ref.save"),
    ("model.module.update_fixed_params()", "model_ref.update_fixed_params()"),
    ("model.module.update_learning_rate()", "model_ref.update_learning_rate()"),
    ("model.module.loss_names", "model_ref.loss_names"),
    # Milestone checkpoint saving (epochs 5, 10, 20, 40)
    (
        "if epoch % opt.save_epoch_freq == 0:",
        "milestone_epochs = [5, 10, 20, 40]\n    if epoch in milestone_epochs or epoch % opt.save_epoch_freq == 0:",
    ),
]
patch_file(os.path.join(BASE, "train.py"), train_replacements, guard="milestone_epochs")


# 2. models/networks.py — make VGGLoss .cuda() conditional on gpu_ids
networks_replacements = [
    (
        "self.vgg = Vgg19().cuda()",
        "self.vgg = Vgg19()\n        if len(gpu_ids) > 0:\n            self.vgg = self.vgg.cuda(gpu_ids[0])",
    ),
]
patch_file(
    os.path.join(BASE, "models/networks.py"),
    networks_replacements,
    guard="self.vgg = self.vgg.cuda(gpu_ids[0])",
)


# 3. models/pix2pixHD_model.py — conditional .cuda() in encode_input / encode_features
model_replacements = [
    (
        "input_label = label_map.data.cuda()",
        "input_label = label_map.data.cuda() if len(self.gpu_ids) > 0 else label_map.data",
    ),
    (
        "input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)",
        "input_label = input_label.scatter_(1, label_map.data.long().cuda() if len(self.gpu_ids) > 0 else label_map.data.long(), 1.0)",
    ),
    (
        "inst_map = inst_map.data.cuda()",
        "inst_map = inst_map.data.cuda() if len(self.gpu_ids) > 0 else inst_map.data",
    ),
    (
        "real_image = Variable(real_image.data.cuda())",
        "real_image = Variable(real_image.data.cuda() if len(self.gpu_ids) > 0 else real_image.data)",
    ),
    (
        "feat_map = Variable(feat_map.data.cuda())",
        "feat_map = Variable(feat_map.data.cuda() if len(self.gpu_ids) > 0 else feat_map.data)",
    ),
    (
        "inst_map = label_map.cuda()",
        "inst_map = label_map.cuda() if len(self.gpu_ids) > 0 else label_map",
    ),
    (
        "image = Variable(image.cuda(), volatile=True)",
        "image = Variable(image.cuda() if len(self.gpu_ids) > 0 else image, volatile=True)",
    ),
    (
        "feat_map = self.netE.forward(image, inst.cuda())",
        "feat_map = self.netE.forward(image, inst.cuda() if len(self.gpu_ids) > 0 else inst)",
    ),
]
patch_file(
    os.path.join(BASE, "models/pix2pixHD_model.py"),
    model_replacements,
    guard="if len(self.gpu_ids) > 0 else label_map.data",
)

print("All patches applied.")
