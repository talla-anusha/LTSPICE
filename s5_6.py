
import re
import numpy as np
import matplotlib.pyplot as plt

# ---------- EXPERIMENT 6: WIDTH VARIATION ----------
EXP = "S5_6"
PARAM = "W (um)"
SCALE = 1e6       # Convert width from metres to micrometres

W_um = 1.0        # Nominal width for threshold criterion
L_um = 0.065      # Nominal channel length in micrometres

MULT = {
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "µ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15
}


def parse_val(s):
    s = s.strip().lower()

    for suf in ("meg", "k", "m", "u", "µ", "n", "p", "f"):
        if s.endswith(suf):
            return float(s[:-len(suf)]) * MULT[suf]

    return float(s)


def load(fname):
    """Return (parameter value, x array, absolute drain current) for each step."""
    curves = []
    val = None
    rows = []

    with open(fname, encoding="latin-1") as f:
        f.readline()  # Skip header

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("Step Information"):
                if rows:
                    a = np.array(rows)
                    curves.append((val, a[:, 0], np.abs(a[:, 1])))
                    rows = []

                match = re.search(r"=\s*([^\s]+)", line)

                if match:
                    val = parse_val(match.group(1))

            else:
                p = line.replace(",", " ").split()
                rows.append([float(p[0]), float(p[1])])

    if rows:
        a = np.array(rows)
        curves.append((val, a[:, 0], np.abs(a[:, 1])))

    return curves


# ---------- LOAD EXPORTED DATA ----------
vg_curves = load(f"{EXP}_vg.txt")
vd_curves = load(f"{EXP}_vd.txt")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

# Fixed nominal-width reference current for VTH extraction
icrit = 100e-9 * W_um / L_um

print(
    f"{'W (um)':>10s} {'VTH (V)':>8s} "
    f"{'ION (uA/um)':>12s} {'IOFF (nA/um)':>14s}"
)

# ---------- ID-VG CURVES AND EXTRACTION ----------
for val, vg, idd in vg_curves:

    lab = f"{val * SCALE:g}"

    # Actual width for this stepped curve, converted to micrometres
    W_curve_um = val * 1e6

    if W_curve_um <= 0:
        raise ValueError(f"Invalid device width: {W_curve_um} um")

    # Normalize current by the actual width of each curve
    j = idd / W_curve_um * 1e6

    ax[0].plot(vg, j, label=lab)

    positive = j > 0
    ax[1].semilogy(vg[positive], j[positive], label=lab)

    # Constant-current threshold criterion uses nominal width
    vth = np.interp(icrit, idd, vg)

    # ON and OFF currents normalized by actual width
    ion = np.interp(1.1, vg, idd) / W_curve_um * 1e6
    ioff = np.interp(0.0, vg, idd) / W_curve_um * 1e9

    print(
        f"{lab:>10s} {vth:8.3f} "
        f"{ion:12.1f} {ioff:14.4f}"
    )


# ---------- ID-VD CURVES ----------
for val, vd, idd in vd_curves:

    lab = f"{val * SCALE:g}"
    W_curve_um = val * 1e6

    ax[2].plot(
        vd,
        idd / W_curve_um * 1e6,
        label=lab
    )


# ---------- GRAPH FORMATTING ----------
ax[0].set(
    xlabel="VG (V)",
    ylabel="ID (uA/um)",
    title="ID-VG, VDS = 0.4 V (linear)"
)

ax[1].set(
    xlabel="VG (V)",
    ylabel="ID (uA/um)",
    title="ID-VG, VDS = 0.4 V (log)"
)

ax[2].set(
    xlabel="VD (V)",
    ylabel="ID (uA/um)",
    title="ID-VD, VGS = 1.1 V"
)

for a in ax:
    a.grid(True, which="both", alpha=0.3)
    a.legend(title=PARAM)

plt.tight_layout()
plt.savefig(f"{EXP}.png", dpi=300)
plt.show()
