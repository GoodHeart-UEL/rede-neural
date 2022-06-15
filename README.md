# Rede Neural
Este repositório é destinado aos códigos referente a rede neural desenvolvida em Python.

## Instalação de dependências
Para realizar a instalação das dependências é necessário ter o ```pip``` instalado. Para instalar as dependências, execute os comandos abaixo:

### Bibliotecas úteis
```
pip install numpy matplotlib xlsxwriter pandas seaborn
```

### Bibliotecas de Machine Learning
```
pip install sklearn tensorflow keras
```

### Biblioteca para manipular arquivos da [PhysioNet](https://physionet.org/)
```
pip install wfdb
```

## Arquivos

 - [GeneratePQRST.py](./GeneratePQRST.py): Script responsável por gerar pontos do complexo QRS a partir do ponto R e uma janela fixa para determinar os pontos Q e S. Gera arquivos (.xlsx) com a posição dos pontos.
 - [ViewPQRST.py](./ViewPQRST.py): Script responsável por gerar gráficamente o ECG com os pontos marcados.
 - [GenerateDataset.py](./GenerateDataset.py): Script responsável por gerar o arquivo ```dataset.csv``` responsável pelo treinamento e teste da rede neural.
 - [DataAnalysis.py](./DataAnalysis.py): Script responsável por gerar gráficos referentes ao cruzamento das características e seus agrupamentos (dispersão).
 - [BinaryNeuralNetwork.py](./BinaryNeuralNetwork.py): Script responsável pela criação e execução do treinamento da rede neural.
