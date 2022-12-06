import pandas as pd

df = pd.read_csv("imports-exports-commerciaux.csv", sep=";")

# On reformate la date et l'heure en combinant les colonnes "date" et "position" (qui représente l'heure) dans une seule colonne timestamp


def heure(n):
    if n < 10:
        return f"0{int(n)}"
    else:
        return str(int(n))


df["timestamp"] = df["date"] + "-" + df["position"].map(heure)

# On enlève les colonnes qui ne nous intéressent pas, c'est à dire toutes celles concernant l'échange spécifique entre la france et un seul autre pays, étant donné que les colonnes imports et exports sont la somme de ces valeurs
df = df.drop(["fr_gb", "gb_fr", "fr_cwe", "cwe_fr", "fr_ch", "ch_fr",
             "fr_it", "it_fr", "fr_es", "es_fr", "date", "position"], axis=1)

df = df.sort_values(by="timestamp")

print(df)
