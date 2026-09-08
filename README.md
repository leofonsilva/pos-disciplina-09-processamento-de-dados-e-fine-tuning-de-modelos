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
