# Projeto de Machine Learning Preditivo

Este projeto demonstra como construir um modelo de Machine Learning preditivo utilizando RandomForest com as bibliotecas `scikit-learn` e `pandas` em Python.

## Objetivo do Projeto

Desenvolver um modelo preditivo que possa prever [inserir aqui a variável alvo, como por exemplo, preço de uma casa, classificação de uma doença, etc.].

## Requisitos

- Python 3.6 ou superior
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib` (opcional, para visualizações)

## Instalação

1. Clone o repositório:
    ```sh
    git clone https://github.com/seu-usuario/seu-repositorio.git
    ```
2. Navegue até o diretório do projeto:
    ```sh
    cd seu-repositorio
    ```
3. Crie e ative um ambiente virtual:
    ```sh
    python -m venv venv
    source venv/bin/activate # Para Linux/Mac
    venv\Scripts\activate # Para Windows
    ```
4. Instale as dependências:
    ```sh
    pip install -r requirements.txt
    ```

## Estrutura do Projeto

```plaintext
├── data
│   └── dataset.csv          # Conjunto de dados
├── notebooks
│   └── data_preprocessing.ipynb  # Notebook para pré-processamento de dados
├── src
│   ├── train.py             # Script para treinar o modelo
│   └── predict.py           # Script para fazer previsões
├── README.md
├── requirements.txt         # Arquivo de dependências
└── .gitignore


O pré-processamento de dados inclui:

Leitura e limpeza dos dados com pandas

Tratamento de valores ausentes

Codificação de variáveis categóricas

Divisão do dataset em conjuntos de treino e teste

import pandas as pd
from sklearn.model_selection import train_test_split

# Leitura dos dados
data = pd.read_csv('data/dataset.csv')

# Limpeza e tratamento dos dados
data = data.dropna()

# Codificação de variáveis categóricas, se necessário
data = pd.get_dummies(data)

# Divisão dos dados
X = data.drop('target_column', axis=1)
y = data['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Treine o modelo utilizando RandomForestClassifier ou RandomForestRegressor dependendo da sua variável alvo (classificação ou regressão).

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Inicialize o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treine o modelo
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


Use o modelo treinado para fazer previsões em novos dados.

new_data = pd.read_csv('data/new_data.csv')
new_data = pd.get_dummies(new_data)
predictions = model.predict(new_data)
print(predictions)


