"""Chat independente por pergunta; execute: python src/chat.py."""
import sys
from search import search_prompt


def main():
    try:
        answer = search_prompt()
    except Exception as error:
        print(f"Não foi possível iniciar o chat: {error if isinstance(error, ValueError) else type(error).__name__ + '. Verifique banco, credenciais e ingestão.'}", file=sys.stderr)
        return 1
    print("Faça sua pergunta (digite sair para encerrar):")
    while True:
        try:
            question = input("PERGUNTA: ").strip()
            if question.casefold() in {"sair", "exit", "quit"}:
                return 0
            if not question:
                continue
            print(f"RESPOSTA: {answer(question)}\n")
        except (EOFError, KeyboardInterrupt):
            print("\nChat encerrado.")
            return 0
        except Exception as error:
            print(f"Erro na consulta ({type(error).__name__}). Verifique conexão, limites e acesso ao modelo; tente novamente.", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
