import numpy as np
import json
import os
from datetime import datetime

class HMM:

  def __init__(self, data, trainPerc, checkpoint, estados):
    self.checkpoint = checkpoint
    trainSize = int(len(data) * trainPerc)
    self.train = data[:trainSize]
    self.tests = data[trainSize:]
    self.estados = estados
    self.observacoes = ['<unk>']

  def inicializaEstados(self):
    self.Pi = {} # initial state distribution
    self.A = {}  # state transition matrix
    self.B = {}  # output distribution
    self.CountPi = 0
    self.CountA = {}
    self.CountB = {}

    for estado in self.estados:
      #Pi
      self.Pi[estado] = 0
      #A
      self.A[estado] = {}
      self.CountA[estado] = 0
      for std in self.estados:
        self.A[estado][std] = 0
      #B
      self.B[estado] = {}
      self.CountB[estado] = 0
      for obs in self.observacoes:
        self.B[estado][obs] = 0

  def atualizaValores(self, pi, A, B):
    countPi = 0
    for qtd in pi.values():
      countPi += qtd
    for estado in self.estados:
      #Pi
      self.Pi[estado] = pi[estado] / countPi
      if countPi == 0:
        countPi = 1
      #A
      countA = 0
      for qtd in A[estado].values():
        countA += qtd
      if countA == 0:
        countA = 1
      for subState in A[estado]:
        self.A[estado][subState] = A[estado][subState] / countA
      #B
      countB = 0
      for qtd in B[estado].values():
        countB += qtd
      if countB == 0:
        countB = 1
      for observacao in B[estado]:
        self.B[estado][observacao] = B[estado][observacao] / countB

  def fit(self):
    
    t0 = datetime.now()
    count = 0
    #costs = []
    self.inicializaEstados()
    Pi = self.Pi.copy()
    A = self.A.copy()
    B = self.B.copy()
    
    for linha in self.train:
      estadoAnterior = 0
      for estado, obs in linha:
        obs = self.verificaVocabulario(obs)
        if estadoAnterior == 0:
          Pi[estado] += 1
        else:
          A[estadoAnterior][estado] +=1
                
        B[estado][obs] +=1
        estadoAnterior = estado
      count+=1

      if count % self.checkpoint == 0:
        print("Linha", count)
        #self.atualizaValores(Pi, A, B)
        #costs.append(self.cost())

    self.atualizaValores(Pi, A, B)     
      
    print("A:", self.A)
    #print("B:", self.B)
    print("pi:", self.Pi)

    print("Fit duration:", (datetime.now() - t0))
    #plt.plot(costs)
    #plt.show()

  def atualizaVocabulario(self, observacao):
    if observacao not in self.observacoes:
      self.observacoes.append(observacao)
      for estado in self.B.keys():
        self.B[estado][observacao] = 0

  def predict(self, obs):
    #trata observacoes desconhecidas
    for i in range(len(obs)):
      if obs[i] not in self.observacoes:
        obs[i] = '<unk>'
    
    #Viterbi
    V=[{}]
    for i in self.estados:
      V[0][i]=self.Pi[i]*self.B[i][obs[0]]      
    for t in range(1, len(obs)):
      V.append({})
      for y in self.estados:
        (prob, state) = max((V[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.estados)
        V[t][y] = prob
      opt=[]
      for j in V:
        for x,y in j.items():
          if j[x]==max(j.values()):
            opt.append(x)
    return opt
  
  def cost(self, data):
    acertos = 0
    total = 0
    
    tests = self.separaDadosParaValidacao(data)

    print("Calculating cost <", end = '')
    i = 0
    for realStateList, obsList in tests:
      result = self.predict(obsList.copy())
      for j in range(len(realStateList)):
        if realStateList[j] == result[j]:
          acertos +=1
        total+=1
      i+=1
      if i % self.checkpoint == 0:
        print(".", end = '')

    print(">")
    return acertos/total

  def fullCost(self, data):
    estadosList = self.estados.copy()
    estadosList.append('')
    #Acertou todos os estados da vaga
    acertosVagas = 0
    totalVagas = 0
    #Acertos dos pares de palavra/estado
    acertosPar = 0
    totalPar = 0
    #Lista de acertos de transicao de estado
    acertosTransicao = {}
    totalTransicao = {}
    #Lista de acertos dentro de cada estado
    acertosEstado = {}
    totalEstado = {}

    #Inicializando valores
    for estado in estadosList:
      acertosTransicao[estado] = {}
      totalTransicao[estado] = {}
      if estado != '':
        acertosEstado[estado] = 0
        totalEstado[estado] = 0
      for estado2 in estadosList:
        if estado2 == estado or estado2 == '':
          continue
        acertosTransicao[estado][estado2] = 0
        totalTransicao[estado][estado2] = 0

    
    tests = self.separaDadosParaValidacao(data)

    print("Calculating cost <", end = '')
    for realStates, observacoes in tests:
      result = self.predict(observacoes.copy())

      qtd = len(realStates)
      acertosTemp = 0
      lastState = ''
      for j in range(qtd):
        if realStates[j] == result[j]:
          acertosTemp +=1
          acertosEstado[realStates[j]] += 1
        totalEstado[realStates[j]] += 1
        if lastState != realStates[j]:
          if realStates[j] == result[j]:
            acertosTransicao[lastState][realStates[j]] += 1
          totalTransicao[lastState][realStates[j]] += 1

        lastState = realStates[j]

      if acertosTemp == qtd:
        acertosVagas += 1
      acertosPar += acertosTemp
      totalPar += qtd

      totalVagas+=1
      if totalVagas % self.checkpoint == 0:
        print(".", end = '')

    percTransicao = {}
    percEstado = {}
    for estado in estadosList:
      percTransicao[estado] = {}
      for estado2 in estadosList:
        if estado2 == estado or estado2 == '':
          continue

        if totalTransicao[estado][estado2] == 0:
          percTransicao[estado][estado2] = -1
        else:
          percTransicao[estado][estado2] = acertosTransicao[estado][estado2]/totalTransicao[estado][estado2]
      
      if estado != '':
        if totalEstado[estado] == 0:
          percEstado[estado] = -1
        else:
          percEstado[estado] = acertosEstado[estado]/totalEstado[estado]

    print(">")

    percTotal =  acertosVagas/totalVagas
    perc = acertosPar/totalPar

    self.resultadosFullCost = percTotal, perc, percTransicao, percEstado
    self.exibeFullCost()

  def exibeFullCost(self):
    percTotal, perc, percTransicao, percEstado = self.resultadosFullCost
    
    print("Loss:", perc)
    print("Percentual acertos totais:", percTotal)
    for estado in percTransicao.keys():
      if estado == '':
        print("Loss em estado inicial")
      else:
        print("Loss em", estado, percEstado[estado])
      print("\t", percTransicao[estado])

  def test(self):
    return self.cost(self.tests)

  def predictRandomTest(self):
    tests = self.separaDadosParaValidacao(self.tests)

    n = np.random.randint(0, len(self.tests))
    result = self.predict(tests[n][1])

    print("Observacoes")
    print("\t", tests[n][1])
    print("Previsto")
    print("\t", result)
    print("Real")
    print("\t", tests[n][0])

    return tests[n][1], result, tests[n][0]

  def ExibeMudancasEstadoIncorretas(self, observacoes, previsto, real):
    prevPrevisto = previsto[0]
    prevReal = real[0]
    
    i = 0
    for obs in observacoes:
      if (prevPrevisto != previsto[i] and real[i] != previsto[i]) or (prevReal != real[i] and real[i] != previsto[i]):
        print("Obs:", obs, "/ Previsto:", previsto[i], "/ Real:", real[i])
      prevPrevisto = previsto[i]
      prevReal = real[i]
      i+=1

  def separaDadosParaValidacao(self, data):
    tests = []
    for test in data:
      testState = []
      testObs = []
      for state, obs in test:
        testState.append(state)
        testObs.append(obs)
      tests.append([testState, testObs])

    return tests

  def salvaResultados(self, path):
    data = {}
    data["A"] = self.A
    data["B"] = self.B
    data["Pi"] = self.Pi   
    #data["CountA"] = self.CountA
    #data["CountB"] = self.CountB
    #data["CountPi"] = self.CountPi 
    with open(path, 'w') as fp:
      json.dump(data, fp)

  def carregaResultados(self, path):  
    with open(path, 'r') as fp:
      data = json.load(fp)
    
    self.A = data["A"]    
    self.B = data["B"]    
    self.Pi = data["Pi"]
    #self.CountA = data["CountA"]
    #self.CountB = data["CountB"]
    #self.CountPi = data["CountPi"]

  def salvaDados(self, path):
    data = {}
    data["Train"] = self.train
    data["Tests"] = self.tests
    data["Estados"] = self.estados
    data["Observacoes"] = self.observacoes
    with open(path, 'w') as fp:
      json.dump(data, fp)

  def carregaDados(self, path):  
    with open(path, 'r') as fp:
      data = json.load(fp)
    
    self.train = data["Train"]    
    self.tests = data["Tests"]    
    self.estados = data["Estados"]
    self.observacoes = data["Observacoes"]

  def salvaTestes(self, path):
    data = {}
    data["teste"] = self.resultadosFullCost   
    with open(path, 'w') as fp:
      json.dump(data, fp)

  def carregaTestes(self, path):  
    with open(path, 'r') as fp:
      data = json.load(fp)
    
    self.resultadosFullCost = data["teste"]
    
  def salvaTudo(self, path, createFolder = False):
    if createFolder:
      os.mkdir(path)
      
    self.salvaResultados(path + "/Resultados.json")
    self.salvaDados(path + "/Dados.json")
    self.salvaTestes(path + "/Testes.json")

  def carregaTudo(self, path):  
      
    self.carregaResultados(path + "/Resultados.json")
    self.carregaDados(path + "/Dados.json")
    self.carregaTestes(path + "/Testes.json")