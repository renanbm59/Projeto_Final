#Set current path (necessario?)
#from os import chdir, getcwd
#wd=getcwd()
#chdir(wd)

import tratamentoTexto as TT

voc = TT.GetVocFromScrapy("../Dados/vocabularioWikipediaSujo.json")
TT.SaveVocabulario("/Dados/vocabularioWikipediaLimpo.txt", voc)