import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

W_um = 1.0

def load(fname):
    d = pd.read_csv(fname, sep="\t", encoding="latin-1")
    d.columns = ["vg", "id"]
    d["id"] = d["id"].abs()
    return d

cases = {
    "Baseline (S2)":          ("idvg_0.05t.txt", "idvg_1.1.txt"),
    "UA = UB = 0":            ("s4_idvg_noEnormal_0.05.txt", "noenormal_1.1.txt"),
    "VSAT x 100":             ("vsat_0.05.txt",   "vsat_1.1.txt"),
}

data = {name: (load(a), load(b)) for name, (a, b) in cases.items()}

# ---- Overlay plots ----
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for name, (lo, hi) in data.items():
    for col, (d, vds) in enumerate([(lo, "50 mV"), (hi, "1.1 V")]):
        j = d["id"] / W_um * 1e6                      # uA/um
        ax[0, col].plot(d.vg, j, label=name)
        p = d[j > 0]
        ax[1, col].semilogy(p.vg, p["id"] / W_um * 1e6, label=name)
        ax[0, col].set_title(f"ID-VG, VDS = {vds} (linear)")
        ax[1, col].set_title(f"ID-VG, VDS = {vds} (log)")

for a in ax.flat:
    a.set_xlabel("VG (V)")
    a.set_ylabel("ID (µA/µm)")
    a.grid(True, which="both", alpha=0.3)
    a.legend()

plt.tight_layout()
plt.savefig("S4_overlay.png", dpi=300)

# ---- Numbers for the report ----
icrit = 100e-9 * W_um / 0.065
print(f"{'Case':16s} {'VTH50(V)':>9s} {'ION(uA/um)':>11s} {'IOFF(nA/um)':>12s} {'ION50(uA/um)':>13s}")
for name, (lo, hi) in data.items():
    vth = np.interp(icrit, lo["id"].values, lo.vg.values)
    ion = np.interp(1.1, hi.vg, hi["id"]) / W_um * 1e6
    ioff = np.interp(0.0, hi.vg, hi["id"]) / W_um * 1e9
    ion50 = np.interp(1.1, lo.vg, lo["id"]) / W_um * 1e6
    print(f"{name:16s} {vth:9.3f} {ion:11.1f} {ioff:12.4f} {ion50:13.1f}")

plt.show()