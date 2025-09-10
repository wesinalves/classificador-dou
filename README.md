# classificador-dou
Repositório para o classificador de documentos do DOU (diário oficial da união)
Classificador de documentos do DOU usando técnicas de Deep Learning.
## Descrição
Este projeto tem como objetivo desenvolver um classificador de documentos do Diário Oficial da União (DOU) utilizando técnicas de Deep Learning. O classificador será capaz de categorizar documentos em diferentes classes, facilitando a organização e a busca por informações relevantes.
De acordo com o texto do documento, o classificador pode identificar categorias como:
- Designar
- Nomear
- Exonerar
- Aposentar
- Outras categorias relevantes

## Funcionalidades
- Pré-processamento de texto: Limpeza e normalização dos dados textuais.
- Extração de características: Utilização de embeddings de palavras para representar o texto.
- Treinamento do modelo: Implementação de uma rede neural para classificação de documentos.
- Avaliação do modelo: Métricas para avaliar a performance do classificador.
- Interface simples para testar o classificador com novos documentos.

## Tecnologias Utilizadas
- Python
- TensorFlow/Keras
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- NLTK/Spacy

## Estrutura do Projeto
- `dataset/`: Contém os dados brutos e pré-processados.
- `notebooks/`: Notebooks Jupyter para exploração de dados e experimentação.
- `src/`: Código-fonte do projeto, incluindo scripts de pré-processamento, treinamento e avaliação do modelo.
- `models/`: Modelos treinados e checkpoints.
- `results/`: Resultados e métricas de avaliação do modelo.
- `README.md`: Documentação do projeto.


## Como Usar
1. Clone o repositório:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   ```  
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare os dados:
   - Coloque os dados do DOU na pasta `dataset/`.
   - Execute o script de pré-processamento:
   ```bash
   python src/preprocess.py
   ```
4. Treine o modelo:
   ```bash
   python src/train.py
   ```
5. Avalie o modelo:
   ```bash
   python src/evaluate.py
   ```
6. Teste o classificador com novos documentos:
   ```bash
   python src/predict.py --input "Caminho/para/novo/documento.txt"
   ```  

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests. Por favor, siga as diretrizes de contribuição no arquivo `CONTRIBUTING.md`.

## Licença
Este projeto está licenciado sob a Licença Apache 2.0. Veja o arquivo `LICENSE` para mais detalhes.

**Devido a LGPD (Lei Geral de Proteção de Dados), os dados utilizados neste projeto não serão compartilhados.**
