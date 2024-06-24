# Extração de características de imagens utilizando GLCM
<!-- TABLE OF CONTENTS -->

## Equipe
    Gustavo José da Silveira Mello

## Tabela de Conteúdo

- [Tabela de Conteúdo](#tabela-de-conteúdo)
- [Sobre o Projeto](#sobre-o-projeto)
  - [Feito Com](#feito-com)
- [Começando](#começando)
  - [Pré-requisitos](#pré-requisitos)
  - [Estrutura de Arquivos](#estrutura-de-arquivos)
  - [Instalação](#instalação)
  - [Exemplo de uso](#exemplo-de-uso)
  - [Edição](#edição)

<!-- ABOUT THE PROJECT -->

## Sobre o Projeto

Este repositório organiza os códigos necessários para utilização do extrator GLCM de características, no qual estão sendo extraídas características dos dados de um dataset e dividindo-as em treino e teste, de modo a resultar em uma acurácia e matriz de confusão através da alimentação das características extraídas no modelo de aprendizado de máquina SVM.

### Descrição do descritor implementado

#### Matriz de Coocorrência de Níveis de Cinza (GLCM - Gray-Level Co-occurrence Matrix):
A GLCM é um descritor de textura que analisa a distribuição espacial dos níveis de cinza em uma imagem, extraindo características baseadas na frequência de ocorrência de pares de pixels com valores de intensidade específicos a uma distância fixa e orientação definida.

As principais características extraídas pela GLCM incluem:
- **Contraste**:
Mede a intensidade do contraste entre um pixel e seu pixel vizinho em toda a imagem. É útil para identificar regiões com variações de intensidade significativas.

- **Correlação**:
Mede a correlação linear entre os níveis de cinza de pixels pareados. Essa característica ajuda a identificar a linearidade e a similaridade nas variações de textura, diferenciando regiões com padrões estruturados das regiões mais homogêneas.

- **Energia** (Uniformidade ou Angular Second Moment):
Mede a uniformidade ou regularidade dos valores de intensidade, indicando o quão uniforme a textura é.

- **Homogeneidade** (Inverse Difference Moment):
Avalia a proximidade da distribuição dos elementos da GLCM da diagonal da matriz. Uma alta homogeneidade indica que os pixels próximos têm valores de intensidade semelhantes, comum em regiões mais suaves e homogêneas da imagem.

### Feito Com

Lista das principais tecnologias e ferramentas utilizadas no desenvolvimento do projeto:

- [Python](https://www.python.org/) - Linguagem de programação de alto nível conhecida por sua simplicidade e legibilidade. É amplamente utilizada para desenvolvimento web, automação, análise de dados, e aprendizado de máquina.
- [Open CV](https://pypi.org/project/opencv-python/) - É uma bilbioteca para diversos tipos de análises e manipulação em imagens, tais como: leitura, detecção, *tracking*, dentre outros.
- [VirtualEnv](https://virtualenv.pypa.io/en/latest/) - Ferramenta que permite criar ambientes isolados de desenvolvimento Python, ou seja, torna possível a utilização de diversas bibliotecas em um mesmo ambiente sem que haja conflitos entre elas.
- [Scikit-Learn](https://scikit-learn.org/stable/) - É uma biblioteca conhecida por seus recursos voltados para o processamento de imagens, extração de características e aprendizado de máquina.

<!-- GETTING STARTED -->

## Começando

Para começar a utilizar este projeto, é necessário ter alguns pré-requisitos de ambiente.

### Pré-requisitos

1. A utilização do ambiente requer a instalação do Python 3.12. Outras bibliotecas Python são necessárias para o funcionamento correto e estão descritas no arquivo requirements.txt
2. A documentação das principais bibliotecas aqui utilizadas pode ser encontrada nos links fornecidos em [Feito Com](#feito-com)
3. É necessário que seja seguida a estrutura de arquivos descrita em [Estrutura de Arquivos](###Estrutura-de-Arquivos) para o funcionamento dos códigos.


### Estrutura de Arquivos

A estrutura de arquivos deve estar disposta da seguinte maneira no momento de instalação:

```bash
GLCM_Feature_Extraction
├── .git
├── Dataset
│   ├── Classe_1
│   │   └── imagem.png
│   │   └── imagem2.png
│   └── Classe_2
│   │   └── imagem.png
│   │   └── imagem2.png
├── .gitattributes
├── GLCM.py
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

Serão explicados os arquivos e diretórios na seção de [Edição](#edição).

### Instalação
1. Para instalar e utilizar esse projeto, realizar o download do projeto no github disponível publicamente no link: 

```
https://github.com/gustavomello-source/GLCM_Feature_Extraction
```

2. Após o download do projeto, acesse a pasta System do sistema.

```
$ cd GLCM_Feature_Extraction
```

3. Crie e ative a sua virtualenv utilizando os comandos:

- **Para sistemas Unix (Linux/macOS):**
```
virtualenv -p python3 .venv
. .venv/bin/activate
```
- **Para Windows:**
```
python -m venv .venv
.venv\Scripts\activate
```

Para mais detalhes sobre como utilizar o virtualenv, acesse este [link] - (https://virtualenv.pypa.io/en/latest/)

4. Instale as dependências
```
(.venv) ....$ pip3 install -r requirements.txt
```

### Exemplo de uso
1. Treinamento, teste e exibição de resultados salvando a matriz de confusão:

```bash
python3 main.py -d path/to/dataset
```

Certifique-se de que o diretório do conjunto de dados esteja organizado corretamente conforme a estrutura de arquivos descrita anteriormente.

#### Observações
- O nome do diretório do conjunto de dados deve ser utilizado como parâmetro durante a execução do arquivo main.py, junto com qualquer outro caminho de diretório necessário.
- É possível alterar qual modelo de aprendizado de máquina é utilizado através de uma breve modificação na variável **model** da classe do extrator.


### Classificador e acurácia
Utilizou-se como classificador o modelo de aprendizado de máquina SVM (Support Vector Machine).

Além disso, para reprodutibilidade dos resultados, deve-se ressaltar os seguintes valores de variáveis:

- **train_test_split**: essa função utilizou as seguintes variáveis:
  - test_size=0.2
  - random_state=1
- **distances**: essa variável do GLCM demonstrou valor:
  - distances = [1]
- **angles**: essa variável do GLCM demonstrou valores:
  - angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
- **glcm**: a chamada do extrator glcm foi feita com os seguintes parâmetros: 
  - glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

Com isso, obteve-se: **Acurácia: 80.36%**


### Edição

Esta seção descreve brevemente cada diretório e arquivo criado.

- **.git**: Diretório escondido criado pelo github.
- **__pycache__**: Diretório criado na execução do código.
- **Dataset**: Pasta vazia de exemplo na qual as classes serão colocadas
- **.gitattributes**: Arquivo escondido criado pelo github.
- **GLCM.py**: Este é o arquivo que contém o código responsável pela instanciação da classe do extrator de características e todos os processos implementados no pipeline de execução.
- **LICENSE**: Licença do projeto seguindo o padrão MIT.
- **main.py**: Este é o arquivo responsável por executar as chamadas do pipeline de execução do projeto, mais especificamente o treinamento e o teste utilizando o extrator e o modelo de machine learning implementados.
- **README.md**: Este arquivo contém informações sobre o sistema, como instalá-lo e usá-lo.
- **requirements.txt**: Este arquivo contém uma lista de pacotes Python necessários para executar o sistema.