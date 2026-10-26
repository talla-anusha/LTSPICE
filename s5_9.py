import re
import numpy as np
import matplotlib.pyplot as plt

EXP   = "S5_9" # file prefix
PARAM = "NSD (×10^19 cm^-3)" # name shown in the legend
SCALE = 1e-19             # multiply the parameter value by this for the legend
W_um  = 1.0             # device width in um (for current per um)
L_um  = 0.065           # device length in um

MULT = {"meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "µ": 1e-6,
        "n": 1e-9, "p": 1e-12, "f": 1e-15}

def parse_val(s):
    s = s.strip().lower()
    for suf in ("meg", "k", "m", "u", "µ", "n", "p", "f"):
        if s.endswith(suf):
            return float(s[:-len(suf)]) * MULT[suf]
    return float(s)

def load(fname):
    """Returns a list of (parameter_value, x_array, id_array)."""
    curves, val, rows = [], None, []
    with open(fname, encoding="latin-1") as f:
        f.readline()                                   # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Step Information"):
                if rows:
                    a = np.array(rows)
                    curves.append((val, a[:, 0], np.abs(a[:, 1])))
                    rows = []
                m = re.search(r"=\s*([^\s]+)", line)   # first name=value
                val = parse_val(m.group(1))
            else:
                p = line.replace(",", " ").split()
                rows.append([float(p[0]), float(p[1])])
    if rows:
        a = np.array(rows)
        curves.append((val, a[:, 0], np.abs(a[:, 1])))
    return curves

vg_curves = load(f"{EXP}_vg.txt")
vd_curves = load(f"{EXP}_vd.txt")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
icrit = 100e-9 * W_um / L_um

print(f"{'value':>10s} {'VTH(V)':>8s} {'ION(uA/um)':>11s} {'IOFF(nA/um)':>12s}")
for val, vg, idd in vg_curves:
    lab = f"{val*SCALE:g}"
    j = idd / W_um * 1e6
    ax[0].plot(vg, j, label=lab)
    ax[1].semilogy(vg[j > 0], j[j > 0], label=lab)
    vth = np.interp(icrit, idd, vg)
    ion = np.interp(1.1, vg, idd) / W_um * 1e6
    ioff = np.interp(0.0, vg, idd) / W_um * 1e9
    print(f"{lab:>10s} {vth:8.3f} {ion:11.1f} {ioff:12.4f}")

for val, vd, idd in vd_curves:
    ax[2].plot(vd, idd / W_um * 1e6, label=f"{val*SCALE:g}")

ax[0].set(xlabel="VG (V)", ylabel="ID (µA/µm)", title="ID-VG, VDS = 0.4 V (linear)")
ax[1].set(xlabel="VG (V)", ylabel="ID (µA/µm)", title="ID-VG, VDS = 0.4 V (log)")
ax[2].set(xlabel="VD (V)", ylabel="ID (µA/µm)", title="ID-VD, VGS = 1.1 V")
for a in ax:
    a.grid(True, which="both", alpha=0.3)
    a.legend(title=PARAM)

plt.tight_layout()
plt.savefig(f"{EXP}.png", dpi=300)
plt.show()