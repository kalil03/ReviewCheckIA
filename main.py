"""
Análise de Sentimento - Reviews Mercado Livre
Executa todo o pipeline com um único comando.

Uso:
    python3 main.py
    python3 main.py --skip-train
    python3 main.py --fast
"""
import argparse
import subprocess
import sys
import time


STEPS = [
    ("01_preprocess",      "PRÉ-PROCESSAMENTO",       "src/01_preprocess.py",       []),
    ("02_eda",             "ANÁLISE EXPLORATÓRIA",    "src/02_eda.py",              []),
    ("03_prepare_dataset", "PREPARAÇÃO DO DATASET",   "src/03_prepare_dataset.py",  []),
    ("04_train",           "TREINAMENTO",             "src/04_train.py",            []),
    ("05_evaluate",        "AVALIAÇÃO",               "src/05_evaluate.py",         []),
    ("06_market_insights", "INSIGHTS DE MERCADO",     "src/06_market_insights.py",  []),
]


def run_step(name: str, title: str, script: str, extra_args: list[str]) -> bool:
    print(f"\n--- {title} ---")

    cmd = [sys.executable, script] + extra_args
    start = time.time()

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        print(f"[{name}] Concluído em {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"[{name}] Falhou após {elapsed:.1f}s (exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"[{name}] Interrompido pelo usuário")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="🇧🇷 Pipeline completo de Análise de Sentimento",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Pular treinamento e avaliação (só preprocessing + EDA + insights)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Modo rápido: 500 amostras, 1 epoch (para testar o pipeline)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Número de epochs (padrão: 3)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Rodar apenas um step (ex: --only 02_eda)",
    )
    args = parser.parse_args()

    print("Iniciando pipeline NLP...")

    total_start = time.time()
    results = {}

    for step_name, title, script, extra_args in STEPS:
        # Filtrar por --only
        if args.only and step_name != args.only:
            continue

        # Pular treinamento se --skip-train
        if args.skip_train and step_name in ("03_prepare_dataset", "04_train", "05_evaluate"):
            print(f"Pulando {step_name}...")
            results[step_name] = "SKIPPED"
            continue

        # Args extras para treinamento
        step_args = list(extra_args)
        if step_name == "04_train":
            if args.fast:
                step_args.extend(["--max-samples", "500", "--epochs", "1"])
            elif args.epochs:
                step_args.extend(["--epochs", str(args.epochs)])

        success = run_step(step_name, title, script, step_args)
        results[step_name] = "OK" if success else "FAILED"

        if not success and not args.only:
            print(f"Pipeline parado devido a erro em {step_name}")
            break

    # Resumo
    total_elapsed = time.time() - total_start
    print("\nResumo do Pipeline:")

    for step_name, title, _, _ in STEPS:
        status = results.get(step_name, "NOT RUN")
        print(f"  {step_name:25s} {status}")

    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)
    print(f"\nTempo total: {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
