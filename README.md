# Projeto de Edge AI - Processo Seletivo PNAAT

**- Nome:** Samuel Jackson Mesquita Lima  
**- E-mail:** samuel.jacksonjml@gmail.com  
**- Data da entrega:** 26/04/2026 

## 📌 Resumo do Projeto
> O projeto atende as demandas de um sistema de classificação de dígitos manuscritos (0–9) usando redes neurais convolucionais(CNNs) com o TensorFlow, treinado com o dataset MNIST para cumprir o desafio de Edge AI proposto pela plataforma PNAAT.

## 📲 Como executar
**1.** Instale as dependências: `pip install -r requirements.txt`  
**2.** Treine o modelo: `python train_model.py`  
**3.** Otimize o modelo: `python optimize.py`  

## 📂 Arquivos
```
ProcessoSeletivoIA/  
├── .github/  
│   └── workflows/  
│       └── ci.yml            # 🤖 Pipeline de correção automática (NÃO ALTERADO)  
│  
├── .devcontainer/            # 🐳 Dev Container (NÃO ALTERADO)  
│   └── devcontainer.json  
│  
├── train_model.py            # 🏅 Treinamento do modelo  
├── optimize_model.py         # 🎯 Conversão e otimização  
├── requirements.txt          # 📄 Dependências do projeto  
├── model.h5                  # 🧮 Modelo treinado (gerado)  
├── model.tflite              # 🦾 Modelo otimizado (gerado)  
└── README.md                 # 📋 Relatório final do candidato  
```
## 🧩 Composição da Rede Neural para o desafio
A `CNN` foi criada com o propósito de ser simples para atender dispositivos de borda com eficiência. Arquitetura do projeto:
- **1ª Camada convolucional:** 32 filtros, Kernel 3x3
  - **1º MaxPooling:** 2x2
- **2ª Camada convolucional:** 64 filtros, Kernel 3x3
  - **2º MaxPooling:** 2x2
- **Flatten:** para transformar uma matriz em um vetor linear
- **Camada densa intermediária:** 64 neurônios
- **Camada de saída:** 10 neurônios, ativação softmax
- **Técnica de Otimização:** Dynamic Range Quantization
- **Bibliotecas:** TensorFlow 2.x (Keras)
  
## ☑️ Resultados Obtidos
- **Acurácia no conjunto de teste:** `[0.9917]` `(99,14%)`
- **Tamanho do modelo original (`model.h5`):** `[1.5 MB]`
- **Tamanho do modelo otimizado (`model.tflite`):** `[129 kB]`
- **Redução de tamanho:** aproximadamente `[91,4%]`

## 💬 Comentários adicionais
- A ferramenta GitHub Actions validou automaticamente a presença e a correta geração dos arquivos;
- Foram preservados os nomes e estruturas dos arquivos propostos pelo desafio, como `model.h5` e `model.tflite`;
- O treinamento ocorreu sob 5 épocas apenas, como sugerido pelo desafio, afim de manter compatibilidade com o Git HUB Actions e não perder eficiência;
- A opinião pessoal deste destaca a gratidão pela oportunidade de participar do desafio proposto.

## 🔍 Link para o repositório original do desafio
[Repositório base PNAT](https://github.com/pnaat/processoseletivoIA)
