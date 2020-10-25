import scrapy
from scrapy_splash import SplashRequest

class VagasSpyder(scrapy.Spider):
    #scrapy crawl vagas -o vagas.json
    name = "vagas"
    start_urls = [
        "https://www.vagas.com.br/"
    ]

    def parse(self, response):
        links = response.css("#BuscasFrequentes .vg-btn-rounded").xpath("@href").extract()
        for page in links:
            if page is not None:
                print(page)
                #yield response.follow(page, self.abre_pagina_vagas)
                yield response.follow(page, self.expande_pag_vagas)


    def expande_pag_vagas(self, response):
        expande = response.css('#maisVagas')
        page = response.request.url

        yield from self.abre_pagina_vagas(response)

        if len(expande) > 0:
            expande = expande.xpath("@data-url").extract_first()
            yield response.follow(self.start_urls[0] + expande, self.expande_pag_vagas)

    def abre_pagina_vagas(self, response):
        pagVagaList = response.css(".link-detalhes-vaga").xpath("@href").extract()
        for page in pagVagaList:
            if page is not None:
                yield response.follow(page, self.obtem_dados_vaga)

    def obtem_dados_vaga(self, response):
        vagaID = response.css(".job-breadcrumb__item--id::text").extract()

        salario = response.css(".icone-salario+ div span *::text").extract()
        salario = ''.join(salario)

        local = response.css(".info-localizacao::text").extract()

        beneficios = response.css(".benefit-label::text").extract()

        descricao = response.css(".texto *::text").extract()
        descricao = ''.join(descricao)

        empresa = response.css(".job-company-presentation *::text").extract()
        empresa = ''.join(empresa)

        yield{
            'vagaID' : vagaID,
            'salario' : salario,
            'local' : local,
            'beneficios' : beneficios,
            'descricao' : descricao,
            'empresa' : empresa
        }

class VocabularioSpyder(scrapy.Spider):
    #scrapy crawl vocabulario -o vocabulario.json
    name = "vocabulario"
    start_urls = [
        "https://pt.wikipedia.org/wiki/Especial:Aleat%C3%B3ria"
    ]
    custom_settings = {
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter'
    }

    def cleanText(self, text):
        removeTokens = ["\t", "\n", ".", ",", "!", "?", "(", ")", "\"", "\'"]

        if isinstance(text, list):
            if len(text) == 0:
                text = ""
            else:
                text = text[0]

        for token in removeTokens:
            text = text.replace(token, "")

        text = text.lower()
        return text

    def removeRepeticoes(self, text):
        result = []

        for word in text.split(" "):
            if word not in result:
                result.append(word)

        return result

    def parse(self, response):
        for i in range(100000):
            yield response.follow(self.start_urls[0], self.obtem_dados)

    def obtem_dados(self, response):
        texto = response.css("p::text").extract()
        texto = ''.join(texto)
        texto = self.cleanText(texto)
        texto = self.removeRepeticoes(texto)

        yield {
            'Texto': texto
        }