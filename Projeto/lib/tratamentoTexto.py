import json
import random

def cleanText(text):
  removeTokens = ["\t", "\r", "\n", ".", ",", "!", "?", "(", ")", "\"", "\'", "/", ";", ":"]

  if isinstance(text, list):
    if len(text) == 0:
      text = ""
    else:
      text = text[0]

  for token in removeTokens:
    text = text.replace(token, "")
  text = text.lower()
  return text
  
#def RemoveRepeticoesTexto(texto):

#def RemoveRepeticoesLista(lista):

def loadDataset(path, shuffleStateOrder = False):
  with open(path) as json_file:
    temp = json.load(json_file)
  data = []
  for vagaDic in temp:
    vaga = []
    keys = list(vagaDic.keys())
    if shuffleStateOrder:
      random.shuffle(keys)
    for key in keys:
      if key == 'vagaID':
        continue
      text = vagaDic[key]
      text = cleanText(text)
      for word in text.split():
        vaga.append([key, word])
    data.append(vaga)  

  return data
 
def getVocabularioFromDataSet(data):
  voc = []
  for linha in data:
    for estado, obs in linha:
      if obs not in voc:
          voc.append(obs)
  return voc
  
def GetVocFromScrapy(path):
  with open(path) as json_file:
    temp = json.load(json_file)
  vocabulario = []
  for pagina in temp:
    lista = pagina["Texto"]
    for word in lista:
      if word not in vocabulario:
        vocabulario.append(word)

  return vocabulario
  
def SaveVocabulario(path, V):
  with open(path, "w", encoding="utf-8") as f:
    for word in V:
      f.write(word+"\n")
      
#def MesclarDatasetVagas(path1, path2, pathSaida):