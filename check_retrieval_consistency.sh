#!/bin/bash

# -----------------------------
# CONFIG
# -----------------------------
OT_FILE="$HOME/STRAP/data/retrieval_results/metadata-ot-stove-pot_retrieved_dataset.hdf5"
SDTW_FILE="$HOME/STRAP/data/retrieval_results/stove-pot_retrieved_dataset.hdf5"

echo ""
echo "======================================"
echo "  STRAP RETRIEVAL SANITY CHECK"
echo "======================================"
echo ""

# -----------------------------
# Check existence
# -----------------------------
if [ ! -f "$OT_FILE" ]; then
    echo "❌ OT file not found: $OT_FILE"
    exit 1
fi

if [ ! -f "$SDTW_FILE" ]; then
    echo "❌ SDTW file not found: $SDTW_FILE"
    exit 1
fi

echo "✔ Found both HDF5 files"
echo ""

# -----------------------------
# Run Python sanity check
# -----------------------------
python3 - << 'EOF'
import h5py, sys, numpy as np

ot_path = "/home/ubuntu/STRAP/data/retrieval_results/metadata-ot-stove-pot_retrieved_dataset.hdf5"
sdtw_path = "/home/ubuntu/STRAP/data/retrieval_results/stove-pot_retrieved_dataset.hdf5"

print("\n📁 Opening HDF5 files...\n")

f_ot = h5py.File(ot_path, "r")
f_sdtw = h5py.File(sdtw_path, "r")

# ------------------------------------------------------------
# 1) DETECT DEMO GROUP AUTOMATICALLY
# ------------------------------------------------------------
def find_demo_group(f):
    candidates = ["data/demo", "demo", "demos", "traj", "data"]
    for c in candidates:
        if c in f:
            return c
    raise KeyError("No demo group found in file: " + str(list(f.keys())))

demo_group_ot = find_demo_group(f_ot)
demo_group_sdtw = find_demo_group(f_sdtw)

print(f"✔ SDTW demo group: {demo_group_sdtw}")
print(f"✔ OT   demo group: {demo_group_ot}")

# ------------------------------------------------------------
# 2) Count demos
# ------------------------------------------------------------
ot_demos = [k for k in f_ot[demo_group_ot].keys() if k.startswith("demo")]
sdtw_demos = [k for k in f_sdtw[demo_group_sdtw].keys() if k.startswith("demo")]

print("\n--------------------------------------")
print("2️⃣  CHECK DEMO COUNTS")
print("--------------------------------------")

print(f"SDTW demos: {len(sdtw_demos)}")
print(f"OT demos:   {len(ot_demos)}")

if len(sdtw_demos) != len(ot_demos):
    print("❌ MISMATCH: demo count differs!")
else:
    print("✔ Demo count matches")

# ------------------------------------------------------------
# 3) Check metadata exists for each demo
# ------------------------------------------------------------
print("\n--------------------------------------")
print("3️⃣  CHECK METADATA FOR EACH DEMO")
print("--------------------------------------")

def check_metadata(f, demo_group, demo_list):
    for d in demo_list:
        grp = f[f"{demo_group}/{d}"]
        if "metadata" not in grp:
            print(f"❌ Missing metadata for: {d}")
            return False
        meta = grp["metadata"]
        for key in ["cost", "start", "end", "source_file", "source_traj_key"]:
            if key not in meta:
                print(f"❌ Metadata missing field '{key}' in demo {d}")
                return False
    return True

ok_sdtw = check_metadata(f_sdtw, demo_group_sdtw, sdtw_demos)
ok_ot   = check_metadata(f_ot,   demo_group_ot,   ot_demos)

if ok_sdtw and ok_ot:
    print("✔ All metadata fields exist in both files")
else:
    print("❌ Metadata inconsistencies detected")

print("\n🎉 DONE!")
EOF
