# Age-Gender-Detector

O **Age-Gender-Detector** é uma aplicação Python que detecta se uma pessoa está olhando para a câmera, e, ao detectar, estima sua idade e gênero com base em redes neurais pré-treinadas.

## Funcionalidades

- **Detecção de olhar**: Identifica se uma pessoa está olhando diretamente para a câmera.
- **Predição de Gênero**: Estima se a pessoa é homem ou mulher.
- **Predição de Idade**: Fornece uma estimativa da faixa etária da pessoa.

## Tecnologias Utilizadas

- **Python**: Linguagem de programação utilizada para o desenvolvimento do projeto.
- **OpenCV**: Biblioteca poderosa para processamento de imagens e vídeos.
- **Dlib**: Usada para a detecção de faces e reconhecimento facial.
- **Redes Neurais**: Modelos de deep learning para predição de idade e gênero.
- **PyInstaller**: Ferramenta utilizada para gerar o executável standalone.

## Como Usar

### 1. Baixando o Executável

Você também pode baixar a versão mais recente do executável na aba [Releases](https://github.com/arthur-cristo/Age-Gender-Detector/releases/latest). Após baixar e descompactar a pasta `Age-Gender-Detector.rar`, você encontrará os seguintes arquivos:
- Age-Gender-Detector.exe - O arquivo executável.
- models/ - A pasta que contém os modelos de treinamento usados pelo detector.

### Executando via Executável
1. Baixe o arquivo `.zip` na seção de [Releases](https://github.com/arthur-cristo/Age-Gender-Detector/releases/latest).
2. Extraia o conteúdo da pasta.
3. Certifique-se de que o arquivo `Age-Gender-Detector.exe` esteja na mesma pasta que a pasta `models/`, que contém os modelos de treinamento.
4. Execute o arquivo `Age-Gender-Detector.exe` para iniciar a aplicação.
### 2. Clonando o Repositório

Outra maneira de obter o projeto é clonando o repositório diretamente do GitHub:

```bash
git clone https://github.com/arthur-cristo/Age-Gender-Detector.git
```
Ou baixando o código-fonte compactado também disponível na seção de [Releases](https://github.com/arthur-cristo/Age-Gender-Detector/releases/latest).
### Executando via Código Python
1. Certifique-se de ter as dependências necessárias instaladas (OpenCV, Dlib, etc.).
```bash
pip install opencv-python dlib numpy 
```
2. Rode o script diretamente no seu ambiente Python.
```bash
python OpenCV.py
```