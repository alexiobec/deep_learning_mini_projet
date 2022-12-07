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

# Teste si toutes les valeurs sont présentes
Liste = sorted(df["timestamp"].tolist())
i = 0
booleen = False
while i < len(Liste)-1 and not booleen:
    if Liste[i] == Liste[i+1]:
        print(Liste[i])
        booleen = True
    i += 1
print("Il y a des doublons dans les dates : "+str(booleen))
# On calcule le nombre de dates attendues,
# comme on sait qu'il n'y a pas de doublons,
# si la taille de la liste est le nombre de valeurs attendues,
# alors il ne manque aucune données
j_par_mois = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
nbr_29_fevrier = 4  # nombre d'années bissextiles entre 2005 et 2022
nbr_valeurs_attendu = (sum(j_par_mois)*(2022-2005)+nbr_29_fevrier)*24

print(
    f"nombre de dates attendues : {nbr_valeurs_attendu}\nnombre de valeurs réel : {len(Liste)}")
