from datetime import datetime

from .HMM_Vagas import HMM

class HMM_Vagas_V2(HMM):

  def __init__(self, data, trainPerc, checkpoint, estados, unkPercent):
    super().__init__(data, trainPerc, checkpoint, estados)
    self.unkPercent = unkPercent/100
    
  def atualizaValores(self, pi, A, B):
    #Pi
    for qtd in pi.values():
      countPi += qtd      
    for estado in self.estados:
      if pi[estado] == 0:
        pi[estado] = 1
        countPi += 1
        
    for estado in self.estados:
      #Pi
      self.Pi[estado] = pi[estado] / countPi
      #A
      countA = 0
      for qtd in A[estado].values():
        if qtd == 0:
            qtd = 1
        countA += qtd
      for subState in A[estado]:
        self.A[estado][subState] = A[estado][subState] / countA
      #B
      countB = 0
      for qtd in B[estado].values():
        if qtd == 0:
            qtd = 1
        countB += qtd
      qtdUnk = int(countB * self.unkPercent)
      countB += qtdUnk
      for observacao in B[estado]:
        if observacao == '<unk>':
          self.B[estado][observacao] = qtdUnk
        self.B[estado][observacao] = B[estado][observacao] / countB

  def fit(self, isNewModel = True):
    
    t0 = datetime.now()
    count = 0
    if isNewModel:
      self.inicializaEstados()
      
    Pi = self.Pi.copy()
    A = self.A.copy()
    B = self.B.copy()
    CountPi = self.CountPi.copy()
    CountA = self.CountA.copy()
    CountB = self.CountB.copy()
    
    for linha in self.train:
      estadoAnterior = 0
      for estado, obs in linha:
        self.atualizaVocabulario(obs)
        if estadoAnterior == 0:
          Pi[estado] += 1
        else:
          A[estadoAnterior][estado] +=1
                
        B[estado][obs] +=1
        estadoAnterior = estado
      count+=1

      if count % self.checkpoint == 0:
        print("Linha", count)

    self.atualizaValores(Pi, A, B)     

    print("Fit duration:", (datetime.now() - t0))