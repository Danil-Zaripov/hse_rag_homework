"""Из корня репозитория: python -m eval.evaluator_excel data/test.jsonl --output test_results.xlsx"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

from openpyxl import Workbook, load_workbook

from eval.eval_functions import score_answer, score_context
from eval.rag_example import RAG


def read_items(dataset_path):
    if dataset_path.suffix.lower() == ".jsonl":
        with dataset_path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)
        return

    source = load_workbook(dataset_path, read_only=True, data_only=True)
    try:
        rows = source.worksheets[0].values
        headers = next(rows, ())
        required = {"question_id", "question", "gold_context", "gold_answer"}
        if not required.issubset(headers):
            raise ValueError(
                f"В первом листе не хватает колонок: {required.difference(headers)}"
            )
        for values in rows:
            if all(value is None for value in values):
                continue
            item = dict(zip(headers, values))
            item["gold_context"] = json.loads(item["gold_context"])
            yield item
    finally:
        source.close()


def _write_report(results, output_path: Path) -> None:
    report = Workbook()
    summary = report.active
    if summary is None:
        raise ValueError("summary is none")

    summary.title = "Итоги"
    metrics = ("context_recall", "context_precision", "answer_correctness", "time")
    valid_results = [row for row in results if row["status"] == "ok"]
    summary.append(metrics)
    summary.append(
        [
            mean(row[key] for row in valid_results) if valid_results else None
            for key in metrics
        ]
    )
    details = report.create_sheet("По вопросам")
    columns = list(results[0]) if results else [
        "question_id", "question", "gold_context", "gold_answer", "context",
        "answer", *metrics, "status", "error",
    ]
    details.append(columns)
    for row in results:
        details.append([row.get(key) for key in columns])
        for cell in details[details.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"
    report.save(output_path)


def evaluate(dataset_path, output_path=None, limit=None):
    dataset_path = Path(dataset_path)
    if output_path is None:
        output_path = dataset_path.with_name(dataset_path.stem + "_results.xlsx")
    output_path = Path(output_path)
    if limit is not None and limit < 1:
        raise ValueError("--limit must be a positive integer.")
    if Path(output_path).resolve() == dataset_path.resolve():
        raise ValueError("Выберите для отчёта отдельный файл.")
    log_path = output_path.with_suffix(".log")
    system = RAG()
    results = []
    with log_path.open("w", encoding="utf-8") as log:
        for number, item in enumerate(read_items(dataset_path), start=1):
            if limit is not None and number > limit:
                break
            start = perf_counter()
            print(
                f"[eval] Question {number}"
                + (f"/{limit}" if limit is not None else "")
                + f": {item['question_id']}",
                file=sys.stderr,
                flush=True,
            )
            row = {
                "question_id": item["question_id"],
                "question": item["question"],
                "gold_context": json.dumps(item["gold_context"], ensure_ascii=False),
                "gold_answer": item["gold_answer"],
                "context": "",
                "answer": "",
                "context_recall": None,
                "context_precision": None,
                "answer_correctness": None,
                "time": perf_counter() - start,
                "status": "skipped",
                "error": "",
            }
            try:
                prediction = system.rag(item["question"])
                row["context"] = json.dumps(prediction["context"], ensure_ascii=False)
                row["answer"] = prediction["answer"]
                row.update(score_context(item["gold_context"], prediction["context"]))
                row["answer_correctness"] = score_answer(
                    item["gold_answer"], prediction["answer"]
                )["answer_correctness"]
                row["status"] = "ok"
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                message = (
                    f"Question {number} ({item['question_id']}) skipped: {row['error']}\n"
                )
                print(message, file=sys.stderr, end="", flush=True)
                log.write(message)
                log.flush()
            row["time"] = perf_counter() - start
            results.append(row)
            _write_report(results, output_path)
            print(
                f"[eval] Saved {len(results)} result(s) to {output_path} "
                f"(status={row['status']}).",
                file=sys.stderr,
                flush=True,
            )
    if not results:
        raise ValueError("В датасете нет вопросов.")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path, help="Путь для сохранения Excel-отчёта")
    parser.add_argument(
        "--limit", type=int, help="Обработать только первые N вопросов"
    )
    args = parser.parse_args()
    print(evaluate(args.dataset_path, args.output, args.limit))
