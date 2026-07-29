import torch
import sys

sys.path.insert(0, '.')
from pfnet_inference import PoinTrGenerator

# Create model
model = PoinTrGenerator()
model_keys = set(model.state_dict().keys())

# Load checkpoint and extract state_dict
ckpt = torch.load('PoinTr/ckpts/point_netG90.pth', map_location='cpu')
state_dict = ckpt['state_dict']

# Apply the same cleaning logic as in load_pfnet_generator
cleaned_ckpt = {}
for k, v in state_dict.items():
    ck = k
    if ck.startswith('module.'):
        ck = ck[len('module.'):]
    if ck.startswith('latentfeature.'):
        ck = ck[len('latentfeature.'):]
    cleaned_ckpt[ck] = v

ckpt_keys = set(cleaned_ckpt.keys())

print(f"Model state_dict keys: {len(model_keys)}")
print(f"Checkpoint cleaned keys: {len(ckpt_keys)}")

# Find missing and extra
missing_in_model = ckpt_keys - model_keys
extra_in_model = model_keys - ckpt_keys

if missing_in_model:
    print(f"\nMissing in model ({len(missing_in_model)}):")
    for k in sorted(missing_in_model)[:10]:
        print(f"  {k}")
    if len(missing_in_model) > 10:
        print(f"  ... and {len(missing_in_model)-10} more")
else:
    print("\n✅ No keys missing in model!")

if extra_in_model:
    print(f"\nExtra in model ({len(extra_in_model)}):")
    for k in sorted(extra_in_model)[:10]:
        print(f"  {k}")
    if len(extra_in_model) > 10:
        print(f"  ... and {len(extra_in_model)-10} more")
else:
    print("✅ No extra keys in model!")

# Shape check
print("\n=== Shape check ===")
matched = 0
mismatched = 0
for k in sorted(model_keys & ckpt_keys):
    model_shape = model.state_dict()[k].shape
    ckpt_shape = cleaned_ckpt[k].shape
    if model_shape == ckpt_shape:
        matched += 1
    else:
        mismatched += 1
        print(f"  MISMATCH: {k}: model={model_shape}, ckpt={ckpt_shape}")

print(f"Matched shapes: {matched}")
print(f"Mismatched shapes: {mismatched}")

# Try actual loading
print("\n=== Trying strict load ===")
try:
    model.load_state_dict(cleaned_ckpt, strict=True)
    print("✅ strict=True loading SUCCESS!")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {total_params:,}")
except RuntimeError as e:
    print(f"❌ strict=True loading FAILED: {e}")