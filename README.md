# Classificador de Gatos e Cachorros com Transfer Learning (MobileNet)

Este repositório contém um projeto de classificação de imagens de gatos e cachorros, utilizando a técnica de Transfer Learning com o modelo MobileNet pré-treinado na base de dados ImageNet.

## Descrição do Projeto

Meu objetivo neste projeto foi construir um classificador binário capaz de distinguir entre imagens de gatos e cachorros. Para acelerar o processo de treinamento e alcançar alta precisão com menos dados, **eu utilizei** o **MobileNet**, um modelo de rede neural convolucional (CNN) leve e eficiente, pré-treinado em um vasto conjunto de dados (ImageNet). O Transfer Learning me permitiu adaptar este modelo pré-existente para a minha tarefa específica.

## Tecnologias Utilizadas

*   **Python**
*   **Keras / TensorFlow**: Para construção e treinamento do modelo de Deep Learning.
*   **MobileNet**: Modelo pré-treinado para extração de características.
*   **ImageDataGenerator**: Para aumento de dados e pré-processamento de imagens.
*   **Matplotlib**: Para visualização dos resultados do treinamento.

## Estrutura do Repositório

*   `catsxdogs/`: Contém os conjuntos de dados de treinamento (`training_set`) e teste (`test_set`), além de imagens para predição individual (`single_prediction`).
*   `catsxdogs_mobilenet.h5`: O modelo Keras salvo após o treinamento.
*   `README.md`: Este arquivo.

## Configuração e Instalação

1.  **Clonar o Repositório:**
    ```bash
    git clone https://github.com/rafxys/Transfer-Learning.git
    cd Transfer-Learning
    ```

2.  **Ambiente de Desenvolvimento:**
    É altamente recomendável usar um ambiente virtual (Anaconda/Miniconda ou `venv`).

3.  **Instalar Dependências:**
    ```bash
    pip install tensorflow keras matplotlib numpy pandas
    ```

## Como Usar o Modelo

### 1. Treinamento (caso queira retreinar)

O notebook `catsxdogs_transfer.ipynb` (ou o código Python original) contém todo o fluxo de trabalho:

*   **Carregamento do Modelo Base**: MobileNet pré-treinado sem a camada de topo.
*   **Adição de Camadas Customizadas**: Camadas densas para classificação binária.
*   **Configuração de Data Augmentation**: `ImageDataGenerator` para aumentar a robustez do modelo.
*   **Compilação e Treinamento**: Utilização do otimizador Adam e perda `binary_crossentropy`.

### 2. Predição com Imagens Novas

Após o treinamento, o modelo `catsxdogs_mobilenet.h5` é salvo. Você pode usá-lo para fazer previsões em novas imagens:

```python
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

# Carregar o modelo salvo
model = load_model('catsxdogs_mobilenet.h5')

# Caminho para a imagem de teste
img_path = 'catsxdogs/single_prediction/floyd3.jpg'

# Carregar e pré-processar a imagem
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Fazer a predição
preds = model.predict(x)

# Interpretar o resultado
if preds[0][0] < 0.5:
    result = 'Gato'
    probability = 1 - preds[0][0]
else:
    result = 'Cachorro'
    probability = preds[0][0]

plt.imshow(img)
plt.axis('off')
plt.title(f"O modelo previu: {result} com probabilidade de {probability*100:.2f}%")
plt.show()
```


