import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

W_um = 1.0                      # device width in um
VSAT_LO, VSAT_HI = 0.7, 1.0    # saturation window for the VGS = 1.5 V curve

df = pd.read_csv("idvd_at_diff_vg.csv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

for vgs, g in df.groupby("vgs"):
    g = g.sort_values("vd")
    vd = g.vd.values
    j = g.id.values / W_um * 1e3          # mA/um
    gds = np.gradient(g.id.values, vd)    # S (for the whole device)
    ax[0].plot(vd, j, label=f"VGS = {vgs:g} V")
    ax[1].plot(vd, gds / W_um * 1e3, label=f"VGS = {vgs:g} V")   # mS/um

ax[0].set(xlabel="VD (V)", ylabel="ID (mA/µm)", title="ID-VD family")
ax[1].set(xlabel="VD (V)", ylabel="gds (mS/µm)", title="gds = dID/dVD")
for a in ax:
    a.grid(True, alpha=0.3)
    a.legend()

# ---- ro at the highest VGS, in the saturation region ----
top = df[df.vgs == df.vgs.max()].sort_values("vd")
vd = top.vd.values
gds = np.gradient(top.id.values, vd)
sel = (vd >= VSAT_LO) & (vd <= VSAT_HI)

gds_avg = gds[sel].mean()
ro_avg = W_um / gds_avg / 1e3                           # kOhm.um
ro_end = W_um / np.interp(VSAT_HI, vd, gds) / 1e3       # kOhm.um at VD = 1.1 V

ax[1].axvspan(VSAT_LO, VSAT_HI, color="gray", alpha=0.15)

print(f"VGS = {df.vgs.max():g} V, saturation window {VSAT_LO} to {VSAT_HI} V")
print(f"gds (mean in window) = {gds_avg*1e3/W_um:.4f} mS/um")
print(f"ro  (mean in window) = {ro_avg:.3f} kOhm.um")
print(f"ro (at VD = {VSAT_HI:.1f} V) = {ro_end:.3f} kOhm.um")

plt.tight_layout()
plt.savefig("S3_idvd.png", dpi=300)
plt.show()