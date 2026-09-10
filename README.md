# Pós Disciplina 09 - Processamento de Dados e Fine-Tuning de Modelos

## Introdução
Pendente...

## Módulos

### Módulo 01: Quando Fazer Fine-Tuning (Decision Framework)

#### **Projeto:** [Amplitude Seguros - Decision Framework Tool](module-01)

**Tecnologias utilizadas:**
- **LLM (Large Language Model)** - Motor de raciocínio para fine-tuning
- **AHP (Analytic Hierarchy Process)** - Método de decisão multicritério para ponderação de critérios
- **Monte Carlo** - Simulação de cenários para análise de risco
- **Real Options** - Avaliação do valor de esperar antes de investir em treinamento
- **NPV e Break-even** - Análise financeira de retorno sobre investimento

**Conceitos abordados:**
- **Fine-tuning vs. RAG:** RAG resolve conhecimento novo (recuperação de contexto externo); fine-tuning resolve comportamento, formato, padrão de resposta, tom e estrutura (ajuste dos pesos do modelo). Fine-tuning não deve ser usado para ensinar fatos novos ao modelo.
- **Quatro Condições Necessárias para Fine-tuning:**
  - **Tarefa Estreita e Repetida:** A tarefa precisa ter uma forma previsível e repetitiva (ex: extrair segurado, placa e valor de orçamentos). Sinal vermelho quando a natureza da tarefa muda a cada chamada.
  - **Alternativas Mais Baratas Esgotadas:** Prompt engineering, RAG, roteamento de modelo e cache já foram testados seriamente antes de considerar treinamento.
  - **Dados Suficientes, Diversos e Representativos:** Exemplos reais cobrindo a variação real da tarefa. Não basta volume bruto; é necessário representatividade.
  - **Tarefa Estável:** O contrato de saída não muda com frequência. Mudanças frequentes exigem retreino constante e custo de manutenção elevado.
- **Gate de Governança:** Verificação binária anterior às quatro perguntas: os dados podem ser processados por um provedor de fine-tuning do ponto de vista legal e de compliance? Dados sensíveis (ex: saúde) exigem atenção adicional. Se o gate falha, o caso não avança, independentemente das outras condições.
- **AHP (Analytic Hierarchy Process):** Método de comparação pareada para derivar pesos dos critérios. A dimensão de dados recebe maior peso (aproximadamente 0,455) porque dados representativos não podem ser criados por decisão imediata da equipe. Razão de consistência é calculada para validar o julgamento.
- **Análise Financeira:** NPV (valor presente dos benefícios futuros menos investimento) e break-even (mês em que o retorno cobre o investimento). Simulação de Monte Carlo (10.000 execuções) transforma premissas incertas em distribuição de resultados possíveis.
- **Real Options:** Quando a única reprovação é insuficiência de dados (reprovação temporária), calcula-se o valor de esperar (opção de treinar mais tarde) utilizando árvore binomial. A decisão de esperar pode ter valor econômico positivo quando o risco de dados insuficientes supera a economia esperada.
- **Tipos de Fine-Tuning:** Full Fine-Tuning (ajusta todos os parâmetros, mais caro), LoRA (adaptadores menores, modelo base congelado, reduz custo e memória), QLoRA (LoRA com quantização em 4 bits). O Decision Framework aprova se vale a pena treinar, não define como treinar (segunda decisão).

**Aplicação prática:**
No contexto da Amplitude Seguros, três casos são analisados com o Decision Framework: Amplitude Auto (extração de segurado, placa e valor de orçamentos de oficina) passa em todas as quatro perguntas e recebe recomendação positiva para fine-tuning; Amplitude Saúde Empresarial (extração de beneficiário, procedimento e valor de recibos médicos) reprova por insuficiência de dados representativos (score 0,35 abaixo do limiar 0,6), recebendo recomendação de "esperar 9 meses" com Real Options; Atendimento ao Cliente (negociação de contestações em conversas abertas) reprova por tarefa aberta e instável, sem valor de espera. O código executa os três casos, aplica o gate de governança, deriva pesos por AHP (razão de consistência ~0,038), calcula NPV positivo para Auto com break-even no 10º mês, executa Monte Carlo, e aplica Real Options apenas para Saúde Empresarial. O framework também demonstra que score ponderado não compensa condições essenciais reprovadas - Atendimento ao Cliente pode ter score maior que Saúde Empresarial, mas é rejeitado por tarefa aberta e instável.

**Comandos executados:**
```bash
cd module-01
node decision-framework-tool.js
node grpo-verifiable-reward-demo.js
```

### Módulo 02: Preparação de Datasets para Fine-Tuning

#### **Projeto:** [Amplitude Seguros - Dataset Pipeline](module-02)

**Tecnologias utilizadas:**
- **Tesseract** - Motor de OCR open source para extração de texto de imagens
- **MinHash** - Técnica de aproximação de similaridade para deduplicação
- **LSH (Locality Sensitive Hashing)** - Redução de comparações para encontrar candidatos a duplicatas
- **Entropia de Shannon** - Medida de diversidade de fontes no dataset
- **JSONL** - Formato de armazenamento para exemplos de treinamento

**Conceitos abordados:**
- **Gate de Relevância:** Quatro perguntas antes de qualquer coleta - o documento contém o ground truth da tarefa? Vem do fluxo real de produção? Ajuda a cobrir a variação real? Pode ser utilizado em treinamento do ponto de vista de compliance? As quatro respostas precisam ser verdadeiras; uma falha já rejeita o documento.
- **Data-Centric AI:** Melhorar sistematicamente o dado e estabelecer critérios explícitos para aquilo que entra no treinamento, em vez de assumir que mais volume sempre significa mais qualidade. O trabalho LIMA é citado como exemplo de curadoria superando volume bruto.
- **OCR com Tesseract:** Extração de texto de documentos de apoio (orçamentos de oficina, recibos médicos). Necessário instalar pacote de idioma português para reconhecimento adequado de acentuação, cedilha e termos específicos. A confiança média do OCR é preservada como metadado para futuros gates.
- **Parser Tolerante:** Utiliza expressões regulares flexíveis para reconhecer variações de rótulo (ex: "segurado", "nome do segurado", "segurado do veículo"). Não depende de posição fixa no texto. A flexibilidade é importante porque documentos de diferentes fontes utilizam rótulos distintos.
- **Validação de Esquema:** Campos obrigatórios ausentes ou valores inválidos (zero, negativo) rejeitam o exemplo. Não se deve aceitar extração incompleta como dado de treinamento, pois isso ensina ao modelo que saída incompleta é aceitável.
- **Esquema Canônico:** Quatro campos estáveis - instrução (descrição da tarefa em linguagem natural), entrada (texto bruto do OCR), saída (objeto estruturado do parser), metadata (caso, arquivo de origem, confiança de OCR, validações). Separa preparação de dados do formato específico do provedor.
- **PII Scrubbing:** Redução de informações pessoalmente identificáveis antes do treinamento. CPF validado pelo dígito verificador; nomes ancorados por rótulos conhecidos (segurado, beneficiário). O limite é assumido: um nome solto em texto livre não seria capturado apenas por âncora de rótulo (exige NER mais robusto, como Microsoft Presidio).
- **Deduplicação com MinHash e LSH:** MinHash transforma cada texto em assinatura para aproximar similaridade de Jaccard; LSH divide assinaturas em bandas para gerar candidatos a duplicatas. Redução de 549 comparações (força bruta) para 20 candidatos (96,4% de redução). Recall perfeito nos pares plantados. Refino exato antes da remoção evita tratar documentos com mesmo template como duplicatas.
- **Balanceamento por Temperatura:** Amostragem com alfa = 0,3 suaviza distribuição em direção a participação mais uniforme entre fontes. Restrição de capacidade: nenhuma fonte pode receber mais exemplos do que possui originalmente. Utiliza método do maior resto para converter pesos contínuos em quantidades inteiras.
- **Entropia de Shannon:** Medida de diversidade do dataset. Quanto mais concentrado em poucas fontes, menor a entropia. Número efetivo de fontes (exponencial da entropia) responde a pergunta intuitiva: a quantas fontes igualmente representadas essa distribuição equivale?

**Aplicação prática:**
No contexto da Amplitude Seguros, o pipeline de preparação transforma documentos brutos em exemplos estruturados. O gate de relevância é aplicado a sete candidatos - apenas orçamento de oficina (Auto) e recibo médico (Saúde Empresarial) passam (boletim de ocorrência não entrega os campos, foto não contém texto, transcrição varia demais, prontuário falha em compliance por excesso de dados clínicos, cadastro de beneficiários não é par entrada-saída). O OCR com Tesseract extrai texto com confiança entre 94%-96%. O parser tolerante identifica "segurado", "placa do veículo" e "valor total do reparo" em diferentes layouts. A validação rejeita exemplos sem placa ou com valor zero. O esquema canônico preserva instrução, entrada, saída e metadata. PII scrubbing reduz nome e CPF (validado por dígito verificador). A deduplicação com MinHash/LSH reduz 305 exemplos brutos para 300 após remoção de duplicatas. O balanceamento por temperatura reduz para 200 exemplos finais (120 Auto, 80 Saúde Empresarial). A entropia de Shannon mostra melhora na diversidade: número efetivo de fontes sobe de 5,516 para 5,723 em Auto e de 4,140 para 4,706 em Saúde Empresarial. O dataset final é menor, mas com menos repetição e distribuição mais equilibrada entre fontes - um exemplo concreto de que remover exemplos pode melhorar o dataset.

**Comandos executados:**
```bash
cd module-02
node data-relevance-scoring-tool.js
node extraction-to-jsonl-tool.js
node pii-scrubbing-gate-tool.js
node dataset-cleaning-balancing-tool.js
```
