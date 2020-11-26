from .HMM_Vagas import HMM

class HMM_Vagas_V2(HMM):

  def __init__(self, data, trainPerc, estados, unkPercent, checkpoint = 0):
    super().__init__(data, trainPerc, estados, checkpoint)
    self.unkPercent = unkPercent/100
      
    
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
      qtdUnk = int(totalB * self.unkPercent)
      totalB += qtdUnk
      self.CountB[estado]['<unk>'] = qtdUnk
      for observacao in self.observacoes:
        self.B[estado][observacao] = self.CountB[estado][observacao] / totalB