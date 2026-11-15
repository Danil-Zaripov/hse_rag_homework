# Использование RAG

RAG использует два разных сервиса:

1. `CourseEmbeddings` из лекции обращается к `deepvk/USER-bge-m3` и превращает страницы PDF и вопросы в векторы. Для него нужен `EMB_KEY`.
2. `deepseek-flash` получает вопрос и найденные фрагменты и формирует ответ. Для него нужен `LLM_API_KEY`.

## Установка

Из корня репозитория:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Заполните в `.env`:

```dotenv
EMB_KEY=ключ_эмбеддера_курса
LLM_API_KEY=ключ_LLM
```

Лимиты уже выставлены в `.env.example`: DeepSeek — `LLM_RPM=10`, эмбеддер — 60 запросов в минуту с burst до 20 и максимум 4 одновременными запросами. Программа ограничивает скорость и автоматически ждёт при ответе HTTP 429, поэтому такие ответы не приводят к падению процесса.

Распакуйте отчёты, если папка `data/reports` ещё не существует:

```bash
python -m zipfile -e data/reports.zip data/reports
```

## Построение индекса

```bash
python -m src.main build
```

Для полного пересоздания индекса:

```bash
python -m src.main build --force
```

Команда читает все PDF из `data/reports`, сохраняет номера страниц и отправляет текст страниц в `CourseEmbeddings`. Индекс появится в `index/`.

Текущий этап, количество обработанных документов и чанков, скорость embedding-запросов и примерное оставшееся время печатаются в `stderr`. Это позволяет отличить медленный сетевой embedding-сервис от локального разбора PDF.

## Один вопрос

```bash
python -m src.main ask "Каковы чистые комиссионные доходы Банка ДОМ.РФ за шесть месяцев 2025 года?"
```

Результат содержит:

```json
{
    "answer": "...",
    "context": [{ "doc_id": "domrf_2025_6m", "pages": [5, 14] }]
}
```

## Оценка на train

Скрипт ниже запускает готовый Excel-evaluator на `data/train.jsonl`:

```bash
python -m eval.evaluator_excel data/train.jsonl --output train_results.xlsx
```

Для пробного запуска только первых 30 вопросов:

```bash
python -m eval.evaluator_excel data/train.jsonl --limit 30 --output train_30_results.xlsx
```

После каждого вопроса Excel-файл сохраняется заново. Если отдельный вопрос или ответ судьи завершится ошибкой, он получает статус `skipped`, ошибка записывается в соседний файл `.log`, а остальные вопросы продолжают выполняться. В Excel такие строки не входят в средние метрики.

Оценка ответа использует тот же `deepseek-flash`, тот же `LLM_BASE_URL`, `LLM_API_KEY` и тот же лимит `LLM_RPM=10`, что и основной RAG.

## LangSmith

Чтобы видеть трассы построения индекса и запросов RAG, добавьте в `.env`:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=hse-rag
```

И затем снова выполните `build` или `ask`.
