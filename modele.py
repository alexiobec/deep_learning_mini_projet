from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import math as m
import plotly.graph_objects as go
import sys
import copy
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
df = df.drop(["fr_gb", "gb_fr", "fr_cwe", "cwe_fr", "fr_ch", "ch_fr", "fr_it",
             "it_fr", "fr_es", "es_fr", "date", "position", "import_france"], axis=1)

df = df.sort_values(by="timestamp")

# Teste si toutes les valeurs sont présentes
Liste = sorted(df["timestamp"].tolist())
i = 0
booleen = False
while i < len(Liste)-1 and not booleen:
    if Liste[i] == Liste[i+1]:
        booleen = True
    i += 1
# On calcule le nombre de dates attendues,
# comme on sait qu'il n'y a pas de doublons,
# si la taille de la liste est le nombre de valeurs attendues,
# alors il ne manque aucune données
j_par_mois = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
nbr_29_fevrier = 4  # nombre d'années bissextiles entre 2005 et 2022
nbr_valeurs_attendu = (sum(j_par_mois)*(2022-2005)+nbr_29_fevrier)*24

# Variables globales

dn = 50000.


# Tranformer les données en dataloader

exports = np.array(df["export_france"].tolist())/dn

# On sépare les données en fonction de l'année
#          train : 2005-2017
#          validation : 2018-2019
#          test : 2020-2021

train_nbr = (sum(j_par_mois)*(2018-2005)+nbr_29_fevrier-1)*24

validation_nbr = train_nbr + (sum(j_par_mois)*(2020-2018))*24
test_nbr = validation_nbr + (sum(j_par_mois)*(2022-2020)+1)*24


train = exports[:train_nbr]
validation = exports[train_nbr:validation_nbr]
test = exports[validation_nbr:]


def xety(data):
    data_jour = torch.Tensor(data).view(-1, 24)
    x = data_jour[:-1].view(1, 24, -1)
    y = data_jour[7:].view(1, -1, 24)
    return x, y


trainx, trainy = xety(train)
validx, validy = xety(validation)
testx, testy = xety(test)

# trainx = 1, seqlen, 1
# trainy = 1, seqlen, 1
trainds = torch.utils.data.TensorDataset(trainx, trainy)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)
validds = torch.utils.data.TensorDataset(validx, validy)
validloader = torch.utils.data.DataLoader(validds, batch_size=1, shuffle=False)
testds = torch.utils.data.TensorDataset(testx, testy)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)


class Mod(nn.Module):
    def __init__(self, num_rnn, cnn_size):
        super(Mod, self).__init__()
        nbr_cnn = {"2": 6,  # nombres de layer pour avoir un champ récepteur de 7
                   "3": 3,
                   "4": 2,
                   "7": 1}
        layer = nn.Sequential(nn.Conv1d(24, 24, cnn_size, stride=1),
                              nn.Sigmoid())
        self.cnn = nn.Sequential(*([layer]*nbr_cnn[str(cnn_size)]))
        self.rnn = nn.Sequential(nn.RNN(24, 24, num_layers=num_rnn))

    def forward(self, x):
        # x.shape = (1,351,7) -> (N,Hin,L)
        # cnn needs (N,L,Hin) (B,D,T)(batch,time,dim)
        # 1 724 1
        xx = x.view(1, 24, -1)
        y = self.cnn(xx)
        yy = y.view(1, -1, 24)
        z, h = self.rnn(yy)
        # y.shape = (1,1,346)
        return z


def test(mod):
    mod.train(False)
    totloss, nbatch = 0., 0
    for data in testloader:
        inputs, goldy = data
        haty = mod(inputs)
        loss = crit(haty, goldy)
        totloss += loss.item()
        nbatch += 1
    totloss /= float(nbatch)
    mod.train(True)
    return totloss


def validate(mod):
    mod.train(False)
    totloss, nbatch = 0., 0
    for data in validloader:
        inputs, goldy = data
        haty = mod(inputs)
        loss = crit(haty, goldy)
        totloss += loss.item()
        nbatch += 1
    totloss /= float(nbatch)
    mod.train(True)
    return totloss


def train(mod, nepochs, lr, testdata=True):
    optim = torch.optim.Adam(mod.parameters(), lr=lr)
    tab_train_loss = []
    tab_test_loss = []
    for epoch in range(nepochs):
        if testdata:
            testloss = test(mod)
        else:
            testloss = validate(mod)
        totloss, nbatch = 0., 0
        for data in trainloader:
            inputs, goldy = data
            optim.zero_grad()
            haty = mod(inputs)
            loss = crit(haty, goldy)
            totloss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()
        totloss /= float(nbatch)
        tab_train_loss.append(totloss)
        tab_test_loss.append(testloss)
    return tab_train_loss, tab_test_loss


if __name__ == "__main__":
    crit = nn.MSELoss()
    tab_value = []
    tab_loss = []
    n_epochs = 40
    lr = 0.005
    num_rnn = 2
    nbr_cnn = 3
    fig = go.Figure()
    model = Mod(num_rnn, nbr_cnn)
    trainloss, testloss = train(model, n_epochs, lr=lr)
    fig.add_trace(go.Scatter(y=np.array(trainloss), x=np.arange(len(trainloss)),
                             mode='lines', name=f"Training loss"))
    fig.add_trace(go.Scatter(y=np.array(testloss), x=np.arange(len(testloss)),
                             mode='lines', name=f"Test loss"))
    fig.update_layout(
        title=f"Loss de training et de test",
        xaxis_title="Epoque",
        yaxis_title="Loss"
    )
    fig.show()
