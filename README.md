# Master-P3-DUMb_AI

DUMb AI е система за интелигентно извличане на информация от документи чрез използване на технологии от областта на изкуствения интелект и семантичното търсене. Проектът позволява качване и обработка на текстови документи, генериране на embeddings и извършване на заявки върху индексираното съдържание чрез локално изпълняван езиков модел.

## Основни функционалности

* Регистрация и удостоверяване на потребители
* Качване и обработка на документи
* Поддръжка на TXT и Markdown файлове
* Автоматично разделяне на документи на текстови сегменти (chunks)
* Генериране на embeddings за всеки сегмент
* Съхранение на embeddings във векторна база данни
* Семантично търсене върху индексираните документи
* Клиент-сървър комуникация чрез сокети
* Локално изпълнение на езиков модел чрез LlamaCPP

## Използвани технологии

### Backend

* Python 3.12+
* AsyncIO

### Изкуствен интелект

* LlamaCPP
* LlamaIndex

### База данни

* MongoDB

### Тестване

* Pytest

## Архитектура

Проектът е разделен на няколко основни слоя:

### Client Layer

Отговаря за взаимодействието с потребителя чрез терминален интерфейс.

### Ingestion Layer

Обработва качените документи чрез последователност от стъпки:

1. Парсване на документа
2. Разделяне на текста на chunks
3. Генериране на embeddings
4. Съхранение във векторното хранилище

### Query Layer

Отговаря за изпълнението на потребителски заявки и извличането на релевантна информация от индексираните документи.

### Data Layer

Осигурява достъп до MongoDB и управление на потребителските и векторните данни.

## Структура на проекта

```text
client/
├── ai_client.py
└── tui.py

services/
├── db/
├── ingestion/
├── query/
└── shared/

infra/
└── mongo/

tests/
├── unit/
└── integration/
```

## Инсталация

### Клониране на проекта

```bash
git clone <repository-url>
cd Master-P3-DUMb_AI-dev
```

### Създаване на виртуална среда

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### Инсталиране на зависимостите

```bash
pip install -e .
```

или

```bash
pip install -r requirements.txt
```

## Стартиране на MongoDB

```bash
docker-compose up -d
```

Проверка:

```bash
docker ps
```

## Стартиране на приложението


Стартиране на клиента:

```bash
python client/tui.py
```

## Тестване

Изпълнение на всички тестове:

```bash
python -m pytest
```

Изпълнение само на unit тестове:

```bash
pytest tests/unit
```

Изпълнение само на integration тестове:

```bash
pytest tests/integration
```

Генериране на coverage отчет:

```bash
pytest --cov=. --cov-report=html
```


## Покрити тестови сценарии

* Успешна регистрация и вход в системата
* Неуспешна автентикация
* Дублиране на потребители
* Обработка на празни документи
* Неподдържани файлови формати
* Грешки при работа с базата данни
* Прекалено големи текстови сегменти
* Socket комуникация клиент–сървър
* End-to-end ingestion и query процес

## Бъдещи подобрения

* Поддръжка на PDF документи
* Web интерфейс
* Допълнителни embedding модели
* По-ефективно векторно търсене
* Поддръжка на множество LLM модели
* Контейнеризация чрез Kubernetes



## Local MongoDB

The project uses MongoDB Atlas Local for local development because the RAG
pipeline needs MongoDB Vector Search.

Start the database:

```bash
docker compose up -d mongodb-atlas-local
```

Initialize collections, normal indexes, schema versioning, and the vector
search index:

```bash
mongosh "mongodb://localhost:27018" infra/mongo/init_db.js
```

Connection string:

```txt
mongodb://localhost:27018
```

More details are in `infra/mongo/README.md`.
