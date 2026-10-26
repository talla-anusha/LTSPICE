import numpy as np
import pandas as pd

# Device parameters
W_um = 1.0
L_um = 0.065
VDD = 1.1

# Read the simulation text files directly
d1 = pd.read_csv(
    "idvg_0.05t.txt",
    sep=r"\s+",
    encoding="latin-1",
    names=["vg", "id"]
)

d2 = pd.read_csv(
    "idvg_1.1.txt",
    sep=r"\s+",
    encoding="latin-1",
    names=["vg", "id"]
)

# Clean both datasets
for df in (d1, d2):
    df["vg"] = pd.to_numeric(df["vg"], errors="coerce")
    df["id"] = pd.to_numeric(df["id"], errors="coerce")

    df.dropna(subset=["vg", "id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Use the magnitude of drain current
    df["id"] = df["id"].abs()

    # Sort by gate voltage
    df.sort_values("vg", inplace=True)
    df.reset_index(drop=True, inplace=True)

# Extract arrays
vg1 = d1["vg"].values
id1 = d1["id"].values

vg2 = d2["vg"].values
id2 = d2["id"].values


# --------------------------------------------------
# 1. VTH using maximum transconductance method
#    VDS = 0.05 V
# --------------------------------------------------

gm = np.gradient(id1, vg1)
k = np.argmax(gm)

vth_gm = vg1[k] - id1[k] / gm[k] - 0.05 / 2


# --------------------------------------------------
# 2. VTH using constant-current method
# --------------------------------------------------

icrit = 100e-9 * W_um / L_um

vth_cc_lo = np.interp(icrit, id1, vg1)
vth_cc_hi = np.interp(icrit, id2, vg2)


# --------------------------------------------------
# 3. ION and IOFF at VDS = 1.1 V
# --------------------------------------------------

ion = np.interp(VDD, vg2, id2) / W_um * 1e6
ioff = np.interp(0.0, vg2, id2) / W_um * 1e9


# --------------------------------------------------
# 4. Subthreshold Swing (SS) for both VDS values
# --------------------------------------------------

# SS at VDS = 0.05 V
mask1 = (id1 > icrit * 1e-4) & (id1 < icrit * 0.5)

if mask1.sum() >= 2:
    slope1 = np.polyfit(
        vg1[mask1],
        np.log10(id1[mask1]),
        1
    )[0]

    ss1 = 1e3 / slope1 if slope1 > 0 else float("nan")
else:
    ss1 = float("nan")


# SS at VDS = 1.1 V
mask2 = (id2 > icrit * 1e-4) & (id2 < icrit * 0.5)

if mask2.sum() >= 2:
    slope2 = np.polyfit(
        vg2[mask2],
        np.log10(id2[mask2]),
        1
    )[0]

    ss2 = 1e3 / slope2 if slope2 > 0 else float("nan")
else:
    ss2 = float("nan")


# --------------------------------------------------
# 5. DIBL calculation
# --------------------------------------------------

dibl = (vth_cc_lo - vth_cc_hi) / (1.1 - 0.05) * 1e3


# --------------------------------------------------
# 6. Display all results
# --------------------------------------------------

print("\n========== S2 EXPERIMENT RESULTS ==========")

print("\n--- Threshold Voltage ---")
print(f"VTH (max-gm, VDS = 0.05 V) = {vth_gm:.3f} V")
print(f"VTH (constant-current, 0.05 V) = {vth_cc_lo:.3f} V")
print(f"VTH (constant-current, 1.1 V)  = {vth_cc_hi:.3f} V")

print("\n--- Drain Current ---")
print(f"ION  (VG = 1.1 V, VDS = 1.1 V) = {ion:.2f} uA/um")
print(f"IOFF (VG = 0 V, VDS = 1.1 V)   = {ioff:.2f} nA/um")

print("\n--- Subthreshold Swing ---")
print(f"SS (VDS = 0.05 V) = {ss1:.2f} mV/dec")
print(f"SS (VDS = 1.1 V)  = {ss2:.2f} mV/dec")

print("\n--- Drain-Induced Barrier Lowering ---")
print(f"DIBL = {dibl:.2f} mV/V")

print("\n--- SS Fitting Points ---")
print(f"Points used at VDS = 0.05 V: {mask1.sum()}")
print(f"Points used at VDS = 1.1 V:  {mask2.sum()}")

print("\n============================================")