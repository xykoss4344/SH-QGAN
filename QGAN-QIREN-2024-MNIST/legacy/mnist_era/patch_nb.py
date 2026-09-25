import nbformat

nb = nbformat.read('QGAN_Evaluation.ipynb', as_version=4)

OLD = "generated_atoms = []\nfor img in fake_flat:\n    atoms, _ = view_atoms(img, view=False)\n    generated_atoms.append(atoms)\n\nprint(f\"✅ Generated {len(generated_atoms)} crystal structures for evaluation.\")"

NEW = """generated_atoms = []
failed = 0
for img in fake_flat:
    try:
        atoms, _ = view_atoms(img, view=False)
        generated_atoms.append(atoms)
    except Exception:
        failed += 1

print(f"✅ Generated {len(generated_atoms)} valid crystal structures ({failed} had invalid cell params).")
assert len(generated_atoms) > 0, "All structures invalid — try a later checkpoint."
"""

patched = 0
for cell in nb.cells:
    if 'generated_atoms = []' in cell.source and 'try:' not in cell.source:
        cell.source = cell.source.replace(OLD, NEW)
        # fallback: replace just the loop if exact match failed
        if 'try:' not in cell.source:
            cell.source = cell.source.replace(
                "generated_atoms = []\nfor img in fake_flat:\n    atoms, _ = view_atoms(img, view=False)\n    generated_atoms.append(atoms)",
                "generated_atoms = []\nfailed = 0\nfor img in fake_flat:\n    try:\n        atoms, _ = view_atoms(img, view=False)\n        generated_atoms.append(atoms)\n    except Exception:\n        failed += 1"
            )
        patched += 1

nbformat.write(nb, 'QGAN_Evaluation.ipynb')
print(f"Patched {patched} cell(s). Done.")
