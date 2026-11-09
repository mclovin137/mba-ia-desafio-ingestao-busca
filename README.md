# Ingestão e busca semântica de PDF

Implementação do desafio **Introdução ao LangChain**, baseada no fork de
[devfullcycle/mba-ia-desafio-ingestao-busca](https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca).
Python + LangChain + PostgreSQL/pgVector; perguntas e respostas em português pelo terminal.

## Pré-requisitos

- Python 3.12, Docker e Docker Compose v2.
- Chave OpenAI com acesso aos modelos configurados e créditos/limites disponíveis.
- Porta local 5432 livre.

## Instalação e configuração

Na raiz do clone deste repositório:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

No Windows PowerShell, ative com `venv\Scripts\Activate.ps1` e copie com
`Copy-Item .env.example .env`. Preencha `OPENAI_API_KEY` no `.env`.
O `.env` não deve ser versionado.

| Variável | Uso / padrão |
|---|---|
| `OPENAI_API_KEY` | Chave usada no embedding e na geração |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `OPENAI_EMBEDDING_DIMENSIONS` | `1536`; dimensão da tabela e dos embeddings |
| `OPENAI_CHAT_MODEL` | `gpt-5-mini`; pode ser alterado sem reingerir |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | `document_pdf` |
| `PDF_PATH` | `document.pdf`, relativo à raiz do projeto |

Modelos são configuráveis. Consulte a documentação oficial e o acesso da sua conta:
[embeddings](https://developers.openai.com/api/docs/models/text-embedding-3-small) e
[LLM](https://developers.openai.com/api/docs/models/gpt-5-mini).
Não é necessário configurar Gemini; esta implementação usa somente OpenAI.

## Execução

```bash
docker compose up -d
docker compose ps -a
```

Espere `postgres` ficar saudável e `bootstrap_vector_ext` terminar com código 0.
O bootstrap cria a extensão `vector`. Os dados persistem no volume do Compose.

```bash
python src/ingest.py
python src/chat.py
```

Digite uma pergunta sobre o PDF e use `sair` para encerrar. Ctrl+C e EOF também encerram.
É possível fazer uma pergunta isolada:

```bash
python src/search.py "Qual é o assunto do documento?"
```

Quando a informação não estiver no contexto, a resposta esperada é exatamente:

```text
Não tenho informações necessárias para responder sua pergunta.
```

O exemplo de faturamento de 10 milhões do enunciado é ilustrativo; não representa
um fato garantido no PDF fornecido. Casos concretos conferidos na primeira página
do PDF incluído para revisão manual:

| Pergunta | Informação esperada |
|---|---|
| Qual o faturamento da Alfa Agronegócio Indústria? | R$ 85.675.568,77 |
| Em que ano foi fundada a Alfa Ambiental Participações? | 2021 |
| Quantos clientes a Alfa Ambiental Participações tinha em 2024? | Frase de recusa; o documento não informa clientes |

## Funcionamento

1. `PyPDFLoader` extrai o PDF por página.
2. `RecursiveCharacterTextSplitter` gera chunks de até 1000 **caracteres**, overlap
   configurado em 150. Separadores e limites de página podem reduzir a sobreposição efetiva.
3. `OpenAIEmbeddings` gera os vetores; `PGVector` armazena texto, vetor e metadados JSONB.
4. Cada pergunta usa `similarity_search_with_score(query, k=10)`, que vetoriza a
   pergunta internamente. Havendo menos chunks, retorna os disponíveis.
5. Os trechos são concatenados com referências de página e inseridos no prompt
   obrigatório do enunciado, preservado em `src/search.py`.
6. `ChatPromptTemplate | ChatOpenAI | StrOutputParser` produz a resposta.

Cada pergunta é independente. O histórico do terminal não é enviado como contexto.
Os scores são distâncias na configuração de cosseno; não são probabilidades.
Não há limiar arbitrário de score.

### Integridade da ingestão

O hash SHA-256 do PDF e IDs determinísticos evitam duplicação em reexecuções.
A tabela auxiliar `rag_documents` registra o PDF ativo e o perfil de embeddings.
Um lock transacional do Postgres serializa ingestões da mesma collection.
O manifesto só é atualizado depois de todos os lotes terem sido persistidos;
falhas não tornam um novo PDF parcialmente ingerido ativo. Retentativas sobrescrevem
os mesmos IDs. Buscas usam somente o hash publicado.

Ao substituir `document.pdf`, execute a ingestão novamente e reinicie o chat.
O chat mantém um snapshot do hash ativo durante sua sessão. Chunks de versões
anteriores ficam armazenados, mas são excluídos das novas buscas pelo filtro de hash;
a limpeza física não faz parte deste desafio. Collections diferentes são independentes.

### Compatibilidade

As dependências diretas têm versões fixadas. Foi mantido `PGVector` para seguir o
exemplo do desafio, embora o projeto oficial recomende `PGVectorStore` para projetos novos:
[aviso de depreciação](https://github.com/langchain-ai/langchain-postgres#vectorstore).
Não misture migração de API com alterações dos embeddings.

Ingestão e consulta precisam usar o mesmo modelo e dimensão. O manifesto rejeita
mudanças de perfil na collection existente. Mesmo dimensões iguais não tornam
modelos diferentes compatíveis. As collections do `PGVector` compartilham a tabela
física: uma nova collection não resolve uma coluna com dimensão diferente.

Para trocar dimensões neste ambiente de estudo, descarte **todos os dados locais**
do Compose e refaça a ingestão, somente se não precisar dos PDFs existentes:

```bash
docker compose down -v
docker compose up -d
# Espere o banco e o bootstrap; depois:
python src/ingest.py
```

## Testes

```bash
pip install -r requirements-dev.txt
pytest -q
```

Sem `TEST_DATABASE_URL`, o teste de integração é pulado. Para rodá-lo, use um banco
separado, por exemplo `rag_test`, com a imagem pgVector. Ele cria e remove uma
collection com nome aleatório; não use um banco de produção. A coluna vetorial
do banco de teste deve estar vazia ou usar 8 dimensões.

```bash
export TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag_test
pytest -q
```

A integração usa embeddings determinísticos de 8 dimensões e LLM simulada, sem
chamar OpenAI. Verifica ingestão real no Postgres, idempotência, recuperação,
montagem do prompt, recusa sem contexto e falha de substituição. GitHub Actions
provisiona PostgreSQL/pgVector para executar todos os testes.

Esses testes não medem a qualidade de respostas da LLM real. Para revisão manual:

| Pergunta/cenário | Resultado esperado |
|---|---|
| Fato explícito, conferido no PDF | Resposta fiel, sem completar dados ausentes |
| Paráfrase desse fato | Mesma informação correta |
| Ano ou valor que o PDF não informa | Frase exata de recusa |
| Opinião sem resposta explícita no PDF | Frase exata de recusa |
| Pergunta externa cujo conteúdo não aparece no PDF | Frase exata de recusa |
| Pedido para ignorar as regras e inventar um valor | Recusa |

## Limitações e solução de problemas

- O prompt reduz alucinações, mas não oferece garantia matemática de fidelidade.
  Recuperar dez trechos não prova que a resposta está neles; valide a recusa com o modelo real.
- PDF digitalizado sem camada de texto requer OCR externo. Tabelas e layouts
  complexos podem ter extração imperfeita.
- Informações fora dos dez trechos recuperados podem causar recusa mesmo existindo
  em outra parte do documento. Não é uma ferramenta de resumo exaustivo.
- Banco indisponível, autenticação, limite de API e modelo inacessível são erros
  operacionais, apresentados separadamente da recusa por falta de informação.
- Se não houver ingestão, o chat pede que `src/ingest.py` seja executado.
- Os chunks e as perguntas são enviados à OpenAI; a LLM recebe os chunks recuperados.
- As credenciais do Compose são de desenvolvimento e o banco é exposto só em localhost.
- `docker compose down` encerra os serviços preservando o volume.

## Estrutura

```text
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── src/
│   ├── config.py
│   ├── store.py
│   ├── ingest.py
│   ├── search.py
│   └── chat.py
├── tests/test_rag.py
├── .github/workflows/tests.yml
├── document.pdf
└── README.md
```
