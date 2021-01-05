import numpy as np
import json
from datetime import datetime

class HMM:
  
  def __init__(self, data, trainPerc, estados, checkpoint = 0):
    self.checkpoint = checkpoint
    trainSize = int(len(data) * trainPerc)
    self.train = data[:trainSize]
    self.tests = data[trainSize:]
    self.estados = estados
    self.observacoes = ['<unk>']
    self.valoresInicializados = False

  def inicializaValores(self):
    self.Pi = {} # initial state distribution
    self.A = {}  # state transition matrix
    self.B = {}  # output distribution
    self.CountPi = {}
    self.CountA = {}
    self.CountB = {}

    for estado in self.estados:
      #Pi
      self.Pi[estado] = 0
      self.CountPi[estado] = 0
      #A
      self.A[estado] = {}
      self.CountA[estado] = {}
      for std in self.estados:
        self.A[estado][std] = 0
        self.CountA[estado][std] = 0
      #B
      self.B[estado] = {}
      self.CountB[estado] = {}
      for obs in self.observacoes:
        self.B[estado][obs] = 0
        self.CountB[estado][obs] = 0
    
    return self.CountPi, self.CountA, self.CountB
    
  def atualizaValores(self):
    #Pi    
    totalPi = 0    
    for estado in self.estados:
      if self.CountPi[estado] == 0:
        self.CountPi[estado] = 1
      totalPi += self.CountPi[estado]  
        
    for estado in self.estados:
      #Pi
      self.Pi[estado] = self.CountPi[estado] / totalPi
      #A
      totalA = 0
      for subState in self.estados:
        if self.CountA[estado][subState] == 0:
            self.CountA[estado][subState] = 1
        totalA += self.CountA[estado][subState]
      for subState in self.estados:
        self.A[estado][subState] = self.CountA[estado][subState] / totalA
      #B
      totalB = 0
      for observacao in self.observacoes:
        if self.CountB[estado][observacao] == 0:
            self.CountB[estado][observacao] = 1
        totalB += self.CountB[estado][observacao]
      for observacao in self.observacoes:
        self.B[estado][observacao] = self.CountB[estado][observacao] / totalB

  def fit(self):
    
    t0 = datetime.now()
    checkProgress = len(self.train) // 100
    count = 0
    if not self.valoresInicializados:
      self.inicializaValores()

    print("Calculating fit<", end = '')
    
    for linha in self.train:
      estadoAnterior = 0
      for estado, obs in linha:
        self.atualizaVocabulario(obs)
        if estadoAnterior == 0:
          self.CountPi[estado] += 1
        else:
          self.CountA[estadoAnterior][estado] +=1
                
        self.CountB[estado][obs] +=1
        estadoAnterior = estado
      count+=1
      
      if count % checkProgress == 0:
        print(".", end = '')
      if self.checkpoint > 0 and count % self.checkpoint == 0:
        self.atualizaValores()

    self.atualizaValores()
    print(">")
    print("Fit duration:", (datetime.now() - t0))

  def atualizaVocabulario(self, observacao):
    if observacao not in self.observacoes:
      self.observacoes.append(observacao)
      for estado in self.estados:
        self.CountB[estado][observacao] = 0
        self.B[estado][observacao] = 0

  def predict(self, observacoes):
    #trata observacoes desconhecidas
    obs = observacoes.copy()
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
          if j[x]== max(j.values()):
            opt.append(x)
    return opt
  
  def cost(self, data):
    acertos = 0
    total = 0
    checkProgress = len(data) // 100
    
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
      if total % checkProgress == 0:
        print(".", end = '')

    print(">")
    return acertos/total

  def fullCost(self, data):
    checkProgress = len(data) // 100
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
    VP_Estado = {}
    FP_Estado = {}
    FN_Estado = {}
    totalEstado = {}

    #Inicializando valores
    for estado in estadosList:
      acertosTransicao[estado] = {}
      totalTransicao[estado] = {}
      if estado != '':
        VP_Estado[estado] = 0
        FP_Estado[estado] = 0
        FN_Estado[estado] = 0
        totalEstado[estado] = 0
      for estado2 in estadosList:
        if estado2 == '':
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
          VP_Estado[realStates[j]] += 1
          acertosTransicao[lastState][realStates[j]] += 1
        else:
          FP_Estado[result[j]] += 1
          FN_Estado[realStates[j]] += 1
          
        totalEstado[realStates[j]] += 1
        totalTransicao[lastState][realStates[j]] += 1
        
        lastState = realStates[j]

      if acertosTemp == qtd:
        acertosVagas += 1
      acertosPar += acertosTemp
      totalPar += qtd

      totalVagas+=1
      if totalVagas % checkProgress == 0:
        print(".", end = '')
        
    percTransicao = {}
    precision = {}
    recall = {}
    f1_score = {}
    for estado in estadosList:
      percTransicao[estado] = {}
      for estado2 in estadosList:
        if estado2 == '':
          continue

        if totalTransicao[estado][estado2] == 0:
          percTransicao[estado][estado2] = -1
        else:
          percTransicao[estado][estado2] = acertosTransicao[estado][estado2]/totalTransicao[estado][estado2]
      
      if estado != '':
        if VP_Estado[estado] + FP_Estado[estado] == 0:
          precision[estado] = - 1
        else:
          precision[estado] = VP_Estado[estado] / (VP_Estado[estado] + FP_Estado[estado])
        if VP_Estado[estado] + FN_Estado[estado] == 0:
          recall[estado] = -1
          f1_score[estado] = -1
        else:
          recall[estado] = VP_Estado[estado] / (VP_Estado[estado] + FN_Estado[estado])
          f1_score[estado] = 2 * (precision[estado] * recall[estado]) / (precision[estado] + recall[estado])

    print(">")

    percTotal =  acertosVagas/totalVagas
    acuracia = acertosPar/totalPar

    self.resultadosFullCost = percTotal, acuracia, percTransicao, precision, recall, f1_score
    self.exibeFullCost()

  def exibeFullCost(self):
    percTotal, acuracia, percTransicao, precision, recall, f1_score = self.resultadosFullCost
    
    print("Acuracia:", acuracia)
    print("Percentual acertos totais:", percTotal)
    for estado in percTransicao.keys():
      if estado != '':
        print("Precision em", estado, precision[estado])
        print("Recall em", estado, recall[estado])
        print("F1 Score em", estado, f1_score[estado])
        print("Precisão de acerto de transição de estado em", estado)
        print("\t", percTransicao[estado])
      else:
        print("Precisão de acerto de transição de estado no estado inicial")
        print("\t", percTransicao[estado])
      print()

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
    print("\nReal")
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
    data["CountA"] = self.CountA
    data["CountB"] = self.CountB
    data["CountPi"] = self.CountPi 
    with open(path, 'w') as fp:
      json.dump(data, fp)

  def carregaResultados(self, path):  
    with open(path, 'r') as fp:
      data = json.load(fp)
    
    self.A = data["A"]    
    self.B = data["B"]    
    self.Pi = data["Pi"]
    self.CountA = data["CountA"]
    self.CountB = data["CountB"]
    self.CountPi = data["CountPi"]
    
    self.valoresInicializados = True

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
    
  def salvaTudo(self, path):
      
    self.salvaResultados(path + "/Resultados.json")
    self.salvaDados(path + "/Dados.json")
    self.salvaTestes(path + "/Testes.json")

  def carregaTudo(self, path):  
      
    self.carregaResultados(path + "/Resultados.json")
    self.carregaDados(path + "/Dados.json")
    self.carregaTestes(path + "/Testes.json")