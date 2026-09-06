import os
import glob

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    modified = False

    # Fix torch.device('mps') -> torch.device('cuda' ...)
    device_str_old1 = "device = torch.device('mps')"
    device_str_old2 = "device=torch.device('mps')"
    device_str_new = "device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')"
    
    if device_str_old1 in content:
        content = content.replace(device_str_old1, device_str_new)
        modified = True
    if device_str_old2 in content:
        content = content.replace(device_str_old2, device_str_new)
        modified = True

    # Fix model.to("mps")
    to_mps = 'model=model.to("mps")'
    to_device = 'model=model.to(device)'
    if to_mps in content:
        content = content.replace(to_mps, to_device)
        modified = True

    # Fix map_location=torch.device('mps')
    map_loc_mps = "map_location=torch.device('mps')"
    map_loc_device = "map_location=device"
    if map_loc_mps in content:
        content = content.replace(map_loc_mps, map_loc_device)
        modified = True

    # Fix torch.mps.manual_seed(seed)
    seed_mps = "torch.mps.manual_seed(seed)"
    seed_new = "if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n        torch.mps.manual_seed(seed)"
    if seed_mps in content:
        # replace carefully to keep indentation if possible, but simplest is just:
        # Actually, let's just do a direct replacement, it might misalign but Python accepts it if it's within a block
        content = content.replace("torch.mps.manual_seed(seed)", "if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.manual_seed(seed)")
        modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {filepath}")

for root, _, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            fix_file(os.path.join(root, file))

