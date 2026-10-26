import pandas as pd

files = {
    "idvg_0.05t.txt": "idvg_0.05.csv",
    "idvg_1.1.txt": "idvg_1.1.csv",
}

for src, dst in files.items():
    df = pd.read_csv(src, sep="\t", encoding="latin-1")
    df.columns = ["vg", "id"]        # rename columns to clean names
    df["id"] = df["id"].abs()        # NMOS current from I(Vd) is negative, so flip sign
    df.to_csv(dst, index=False)
    print(src, "->", dst)
    print(df.head(3))
    print("rows:", len(df), "| vg range:", df.vg.min(), "to", df.vg.max())
    print()