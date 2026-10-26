import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

W_um = 1.0   # device width in um

def load(fname):
    d = pd.read_csv(fname, sep=None, engine="python", encoding="latin-1")
    vg = d.iloc[:, 0].to_numpy(dtype=float)
    i = np.abs(d.iloc[:, 1].to_numpy(dtype=float)) / W_um   # A/um
    return vg, i

vg1, id1 = load("idvg_0.05.csv")   # VDS = 50 mV
vg2, id2 = load("idvg_1.1.csv")    # VDS = 1.1 V

print("50 mV: ID(VG=0) =", id1[0], " ID(VG=1.1) =", id1[-1])
print("1.1 V: ID(VG=0) =", id2[0], " ID(VG=1.1) =", id2[-1])

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# ---- linear ----
ax[0].plot(vg1, id1 * 1e6, label="VDS = 50 mV")
ax[0].plot(vg2, id2 * 1e6, label="VDS = 1.1 V")
ax[0].set(xlim=(0, 1.1), xlabel="Gate Voltage VGS (V)",
          ylabel="Drain Current ID (µA/µm)",
          title="S2: Transfer Characteristics (Linear)")

# ---- log ----
m1 = id1 > 1e-13
m2 = id2 > 1e-13
ax[1].semilogy(vg1[m1], id1[m1], label="VDS = 50 mV")
ax[1].semilogy(vg2[m2], id2[m2], label="VDS = 1.1 V")
ax[1].set(xlim=(0, 1.1), ylim=(1e-10, 1e-2),
          xlabel="Gate Voltage VGS (V)",
          ylabel="Drain Current ID (A/µm)",
          title="S2: Transfer Characteristics (Log)")

for a in ax:
    a.grid(True, which="both", linestyle="--", alpha=0.4)
    a.legend()

plt.tight_layout()
plt.savefig("S2_Linear_Log.png", dpi=300)
plt.show()



