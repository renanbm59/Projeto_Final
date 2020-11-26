from datetime import datetime

from .HMM_Vagas import HMM

class HMM_Vagas_V3(HMM):

  def __init__(self, data, trainPerc, estados, vocabulario, checkpoint = 0):
    super().__init__(data, trainPerc, estados, checkpoint)
    self.observacoes = vocabulario
    self.observacoes.append("<unk>")

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
        obs = self.verificaVocabulario(obs)
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
        
  def verificaVocabulario(self, observacao):
    if observacao not in self.observacoes:
      return '<unk>'
    return observacao