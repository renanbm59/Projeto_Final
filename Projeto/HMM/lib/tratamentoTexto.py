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

def SaveDatasetVagas(path, dados):
  with open(path, 'w') as outfile:
    json.dump(dados, outfile)
      
def MesclarDatasetVagas(path1, path2, pathSaida):
  with open(path1) as json_file:
    data1 = json.load(json_file)
  with open(path2) as json_file:
    data2 = json.load(json_file)
        
  for vagaDic in data2:
    if vagaDic not in data1:
      data1.append(vagaDic)  
  
  SaveDatasetVagas(pathSaida, data1)
  
  return data1
 
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

def GetVocFromArqList(path):
  vocabulario = []
  with open(path, "r", encoding="utf-8") as f:
    for palavra in f:
      vocabulario.append(cleanText(palavra))
  
  return vocabulario
      
def LimpaVocabulario(vocabulario):
  vocResult = []
  for palavra in vocabulario:
    palavra = cleanText(palavra)
    if palavra not in vocResult:
      vocResult.append(palavra)
            
  return vocResult
  
def SaveVocabulario(path, V):
  with open(path, "w", encoding="utf-8") as f:
    for word in V:
      f.write(word+"\n")
  print("Salvo com sucesso!")