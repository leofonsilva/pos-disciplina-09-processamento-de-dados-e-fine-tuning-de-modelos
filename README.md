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

### Módulo 03: Fine-Tuning via API (upload e train)

#### **Projeto:** [Amplitude Seguros - Vertex AI Fine-Tuning Pipeline](module-03)

**Tecnologias utilizadas:**
- **Vertex AI** - Plataforma gerenciada de fine-tuning do Google Cloud
- **Gemini 2.5 Flash** - Modelo base utilizado para treinamento supervisionado
- **JSONL** - Formato de dataset para upload e treinamento
- **Google Cloud Storage** - Armazenamento de datasets em bucket
- **IAM (Identity and Access Management)** - Controle de permissões para acesso ao bucket e jobs
- **SHA-256** - Hash de conteúdo para versionamento de dataset
- **Model Card** - Documentação estruturada de linhagem do modelo
- **DPO (Direct Preference Optimization)** - Técnica de Preference Tuning para pares de respostas

**Conceitos abordados:**
- **Fine-Tuning Gerenciado:** Segunda decisão do pipeline (como treinar). Processo em cinco passos genéricos: converter dataset, fazer upload, configurar job, treinar, monitorar. O provedor gerencia infraestrutura, GPU e processo distribuído.
- **Risco de Provedor:** Serviços self-service de fine-tuning podem ser descontinuados (OpenAI, Gemini API). Diligência de provedor é parte da arquitetura: disponibilidade futura, suporte ao modelo base, contrato e estratégia de saída.
- **Conversão de Dataset:** Esquema canônico (instrução, entrada, saída, metadata) é adaptado para o formato do provedor (dois turnos: user e model). Instrução e entrada combinadas no turno do usuário; saída preservada como JSON exato no turno do modelo.
- **Gate de Confiança de OCR:** Exemplos com confiança de OCR abaixo de um limiar (ex: 62%) são sinalizados para revisão humana em vez de irem automaticamente para treinamento. Confiança entre 94%-96% passa; ausência de métrica (dados sintéticos) é tratada como não aplicável.
- **Escala do Pipeline:** Mesmo pipeline do módulo 2 (MinHash, LSH, balanceamento por temperatura, entropia de Shannon) aplicado a 305 exemplos brutos → 300 após deduplicação → 200 finais (120 Auto, 80 Saúde Empresarial). Diversidade efetiva de fontes melhora (Auto: 5,516 → 5,723; Saúde: 4,140 → 4,706).
- **Reavaliação com Decision Framework:** Saúde Empresarial, reprovado no módulo 1 por insuficiência de dados (score 0,35), é reavaliado 9 meses depois. Score sobe para 0,62 (acima do limiar 0,6), volume mensal cresce de 1.200 para 1.862 casos. O mesmo framework que reprovou agora aprova, sem alterar a régua.
- **Upload Versionado:** Cada versão de dataset mantém seu próprio caminho no bucket. Não se sobrescreve silenciosamente o arquivo anterior. Um job que falha ou produz resultado inesperado precisa continuar apontando para exatamente os dados usados naquele treinamento.
- **Estados do Job:** Pending (fila, aguardando recursos), Running (treinamento em execução), Succeeded (concluído com sucesso), Failed (falha com possível causa: formato inválido, cota excedida, hiperparâmetro rejeitado).
- **Hiperparâmetros:** Epoch count (número de passagens pelo dataset), Learning rate multiplier (intensidade do ajuste sobre taxa base interna), Adapter size (capacidade da camada adaptativa). Valores usados: 3 épocas, multiplicador 5, adapter size 4.
- **Validação de Hiperparâmetros:** Primeira camada - validação local antes da chamada de rede (epoch count entre 1-20, learning rate multiplier entre 0,1-10). Segunda camada - comparação entre valores solicitados e efetivamente aplicados pelo provedor.
- **Automação Segura:** Cinco passos encadeados (converter, validar, upload, criar job, acompanhar). Fail fast: falha em uma etapa interrompe o fluxo antes de criar recursos. Confirmação explícita antes da criação do job (ação cobrável). Idempotência parcial: upload é idempotente; criação de job não é.
- **Polling com Backoff Exponencial:** Primeira consulta imediata; intervalos crescem por fator 1,5 até teto de 60 segundos. Callback para tornar o acompanhamento observável (Pending, Running, Succeeded). Injeção de dependência permite testar lógica assíncrona sem rede e sem espera real.
- **Versionamento e Model Card:** Hash SHA-256 do conteúdo do dataset como identificador de versão (não apenas nome do arquivo). Model Card registra job, modelo ajustado, endpoint, modelo base, dataset e hash, hiperparâmetros aplicados, estatísticas, timestamps, custo estimado e custo real.
- **Preference Tuning:** Alternativa ao Supervised Fine-Tuning. Em vez de uma resposta correta, fornece par preferido/rejeitado. Relacionado a DPO (Direct Preference Optimization). Mais adequado quando qualidade é subjetiva (tom, voz de marca, linguagem de compliance) e existe mais de uma resposta plausível.

**Aplicação prática:**
No contexto da Amplitude Seguros, o dataset de 200 exemplos (120 Auto, 80 Saúde Empresarial) é convertido para o formato do Vertex AI, validado (13 testes automatizados), e enviado para um bucket no Google Cloud Storage. O job é configurado com Gemini 2.5 Flash, 3 épocas, learning rate multiplier 5, adapter size 4. A validação local bloqueia epoch count zero antes da chamada de rede (após um incidente real onde a API aceitou o valor inválido e iniciou treinamento). A comparação entre pedido e aplicado confirma os valores. O job real leva aproximadamente 45 minutos e 42 segundos, processa 27.353 tokens, custa entre R$2,39 (billing real) e aproximadamente 5-11 centavos de dólar (estimativa). O endpoint publicado é versionado com hash SHA-256 do dataset e documentado em um Model Card que registra toda a linhagem. O teste com Preference Tuning usa 40 exemplos e leva 17 minutos e 32 segundos, publicando modelo ajustado e endpoint. O pipeline completo é automatizado em JavaScript e Python, com 18 testes cobrindo conversão, validação, upload, criação protegida por confirmação, backoff e loop assíncrono com injeção de dependência.

**Comandos executados:**
```bash
cd module-03
node dataset-upload-and-tracking-tool.js
node m3-dataset-scaling-tool.js
node hyperparameter-and-monitoring-tool.js
node finetuning-automation-tool.js
node model-versioning-tool.js
```
