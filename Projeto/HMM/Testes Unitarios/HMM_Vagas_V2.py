from .HMM_Vagas import HMM

class HMM_Vagas_V2(HMM):

  def __init__(self, data, trainPerc, checkpoint, estados, unkPercent):
    super().__init__(data, trainPerc, checkpoint, estados)
    self.unkPercent = unkPercent/100
    
  def atualizaValores(self):
    #Pi
    totalPi = 0
    for qtd in self.CountPi.values():
      if qtd == 0:
        qtd = 1
      totalPi += qtd      
    for estado in self.estados:
      if self.CountPi[estado] == 0:
        self.CountPi[estado] = 1
        totalPi += 1
        
    for estado in self.estados:
      #Pi
      self.Pi[estado] = self.CountPi[estado] / totalPi
      #A
      totalA = 0
      for qtd in self.CountA[estado].values():
        if qtd == 0:
            qtd = 1
        totalA += qtd
      for subState in self.estados:
        self.A[estado][subState] = self.CountA[estado][subState] / totalA
      #B
      totalB = 0
      for qtd in self.CountB[estado].values():
        if qtd == 0:
            qtd = 1
        totalB += qtd
      qtdUnk = int(totalB * self.unkPercent)
      totalB += qtdUnk
      self.CountB[estado]['<unk>'] = qtdUnk
      for observacao in self.observacoes:
        self.B[estado][observacao] = self.CountB[estado][observacao] / totalB