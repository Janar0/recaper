# recaper — Генератор рекап-видео по манге

**AI-инструмент для автоматического создания рекап-видео из глав манги и манхвы.**

Извлекает панели, анализирует содержимое с помощью LLM, генерирует сценарий, озвучивает и собирает финальное видео.

> **[English version](README.md)**

---

## Содержание

- [Возможности](#возможности)
- [Демо](#демо)
- [Требования](#требования)
- [Установка](#установка)
- [Конфигурация](#конфигурация)
- [Использование](#использование)
- [Пайплайн](#пайплайн-этапы-обработки)
- [Примеры](#примеры)
- [Структура проекта](#структура-проекта)
- [FAQ](#faq--troubleshooting)
- [Лицензия](#лицензия)

---

## Возможности

- **Умное извлечение панелей** — детекция через YOLO с фоллбэком на LLM vision
- **Автоопределение контента** — манга (ч/б, справа-налево), манхва (цвет, вертикальная), маньхуа (цвет, слева-направо)
- **AI-анализ панелей** — действие, персонажи, диалоги, настроение, важность (1-10)
- **Генерация сценария** — связный рекап с настроением и темпом повествования
- **Озвучка (TTS)** — синтез речи через Qwen3-TTS с кастомным голосом
- **Рендеринг видео** — эффект Ken Burns, переходы, размытый фон
- **Возобновляемый пайплайн** — 7 этапов, можно продолжить с места остановки
- **Веб-интерфейс** — управление и мониторинг через FastAPI
- **Мульти-формат** — CBZ, CBR, папки с изображениями

---

## Демо

<!-- TODO: вставить скриншоты/GIF -->

<details>
<summary>Пример вывода CLI</summary>

```
TODO: вставить вывод recaper process
```

</details>

<details>
<summary>Скриншот веб-интерфейса</summary>

```
TODO: вставить скриншот веб-интерфейса
```

</details>

<details>
<summary>Пример финального видео</summary>

```
TODO: вставить ссылку или превью видео
```

</details>

---

## Требования

| Компонент | Минимум | Примечание |
|-----------|---------|------------|
| Python | 3.11+ | Рекомендуется 3.12 |
| ffmpeg | любая актуальная | Должен быть в PATH |
| OpenRouter API | ключ | [Получить здесь](https://openrouter.ai/keys) |
| GPU (CUDA) | опционально | Нужен только для локального TTS (Qwen3-TTS) |
| RAM | 4 GB+ | 8 GB+ если используется TTS |
| Дисковое пространство | ~2 GB | Для моделей YOLO + TTS |

---

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/Janaro/recaper.git
cd recaper
```

### 2. Установка ffmpeg

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Linux (Arch):**
```bash
sudo pacman -S ffmpeg
```

**Windows:**
```
1. Скачать ffmpeg: https://www.gyan.dev/ffmpeg/builds/ (release full)
2. Распаковать архив
3. Добавить путь к папке bin в переменную PATH
4. Проверить: ffmpeg -version
```

Или через пакетные менеджеры:
```powershell
# winget
winget install Gyan.FFmpeg

# scoop
scoop install ffmpeg

# choco
choco install ffmpeg
```

### 3. Создание виртуального окружения

```bash
python -m venv .venv
```

Активация:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (cmd)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 4. Установка пакета

```bash
# Базовая установка (только CLI)
pip install -e .

# С веб-интерфейсом
pip install -e ".[web]"

# С озвучкой (TTS) — нужен GPU
pip install -e ".[tts]"

# С поддержкой RAR (.cbr файлы)
pip install -e ".[cbr]"

# Всё сразу
pip install -e ".[web,tts,cbr]"

# Для разработки
pip install -e ".[web,tts,cbr,dev]"
```

### 5. Настройка окружения

```bash
cp .env.example .env
```

Откройте `.env` и вставьте ваш OpenRouter API ключ:

```env
RECAPER_OPENROUTER_API_KEY=sk-or-v1-ваш-ключ
```

### Проверка установки

```bash
recaper version
```

---

## Конфигурация

Все настройки задаются через переменные окружения (префикс `RECAPER_`) или файл `.env`.

### LLM (OpenRouter)

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `RECAPER_OPENROUTER_API_KEY` | — | **Обязательно.** API ключ OpenRouter |
| `RECAPER_OPENROUTER_MODEL` | `anthropic/claude-sonnet-4-20250514` | Основная модель для анализа и скрипта |
| `RECAPER_OCR_MODEL` | `google/gemini-2.0-flash-001` | Дешёвая vision-модель для OCR |
| `RECAPER_LLM_FALLBACK_MODEL` | = OCR модель | Фоллбэк-модель для детекции панелей |
| `RECAPER_LLM_TEMPERATURE` | `0.7` | Температура генерации |
| `RECAPER_LLM_BATCH_SIZE` | `4` | Панелей на один запрос к LLM |
| `RECAPER_LLM_MAX_RETRIES` | `3` | Повторных попыток при ошибке |
| `RECAPER_LLM_MAX_IMAGE_SIZE` | `1024` | Макс. размер изображения (px) |
| `RECAPER_LLM_JPEG_QUALITY` | `80` | Качество JPEG для LLM (10-100) |

### Детекция панелей

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `RECAPER_PANEL_DETECTOR` | `mosesb/best-comic-panel-detection` | YOLO-модель для панелей |
| `RECAPER_PANEL_CONFIDENCE` | `0.45` | Порог уверенности YOLO |
| `RECAPER_MIN_PANEL_AREA_RATIO` | `0.02` | Мин. площадь панели (доля от страницы) |
| `RECAPER_PANEL_PADDING` | `10` | Отступ при кропе (px) |
| `RECAPER_MIN_PANEL_IMPORTANCE` | `4` | Мин. важность для включения в рекап (1-10) |

### Контент и язык

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `RECAPER_LANGUAGE` | `ru` | Язык повествования |

### TTS (озвучка)

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `RECAPER_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Модель TTS |
| `RECAPER_TTS_SPEAKER` | `ryan` | Пресет голоса |
| `RECAPER_TTS_LANGUAGE` | `Russian` | Язык синтеза речи |
| `RECAPER_TTS_INSTRUCT` | спокойный ведущий | Инструкция для стиля голоса |
| `RECAPER_TTS_MAX_RETRIES` | `2` | Макс. повторов при ошибке TTS |

### Видео

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `RECAPER_VIDEO_FPS` | `30` | FPS выходного видео |
| `RECAPER_VIDEO_WIDTH` | `1920` | Ширина видео |
| `RECAPER_VIDEO_HEIGHT` | `1080` | Высота видео |
| `RECAPER_KEN_BURNS_ZOOM` | `1.05` | Коэффициент зума Ken Burns |
| `RECAPER_TRANSITION_DURATION` | `0.8` | Длительность переходов (сек) |
| `RECAPER_PANEL_PADDING_SEC` | `0.3` | Пауза тишины между сценами (сек) |

---

## Использование

### CLI

#### Обработка манги

```bash
recaper process ./manga.cbz --output ./work --title "Название рекапа" --verbose
```

#### Возобновление прерванной обработки

```bash
recaper process ./manga.cbz --output ./work --resume
```

#### С переопределением модели

```bash
recaper process ./manga.cbz -o ./work -m "google/gemini-2.0-flash-001"
```

#### Фильтрация по важности

```bash
# Только важные панели (6+)
recaper process ./manga.cbz -o ./work --min-importance 6
```

#### Все опции CLI

| Опция | Описание |
|-------|----------|
| `--output, -o` | Рабочая директория (по умолчанию: `./work`) |
| `--title, -t` | Название для повествования |
| `--model, -m` | Переопределение модели OpenRouter |
| `--batch-size` | Панелей на запрос (0 = из конфига) |
| `--resume` | Продолжить с последнего завершённого этапа |
| `--min-importance` | Мин. важность панели 1-10 (по умолчанию: 4) |
| `--verbose, -v` | Подробное логирование |

### Веб-интерфейс

```bash
# Запуск сервера
recaper web --host 0.0.0.0 --port 8000

# Или через run.py
python run.py --host 127.0.0.1 --port 8000

# С авто-перезагрузкой (для разработки)
recaper web --reload
```

После запуска откройте http://localhost:8000 в браузере.

#### REST API эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/api/config` | Текущая конфигурация |
| `GET` | `/api/jobs` | Список всех задач |
| `GET` | `/api/jobs/stats` | Статистика задач |
| `POST` | `/api/jobs` | Создать новую задачу |
| `GET` | `/api/jobs/{id}/status` | Статус задачи |
| `GET` | `/api/jobs/{id}/log` | Логи задачи (streaming) |

---

## Пайплайн (этапы обработки)

recaper обрабатывает мангу через 7 последовательных этапов. Каждый этап можно пропустить при использовании `--resume`.

### 1. Распаковка (Unpack)

Извлекает страницы из CBZ/CBR архива или собирает изображения из директории. Конвертирует в PNG.

```
./work/pages/001.png, 002.png, ...
```

### 2. Определение типа (Detect)

Автоматически определяет тип контента по цветности и пропорциям:
- **Манга** — чёрно-белая, чтение справа-налево
- **Манхва** — цветная, вертикальный скролл
- **Маньхуа** — цветная, чтение слева-направо

### 3. Извлечение панелей (Extract)

Находит и вырезает отдельные панели со страниц:
- Основной метод: YOLO-детекция (для манги/маньхуа) или вертикальная нарезка (для манхвы)
- Фоллбэк: LLM vision, если основной метод не справился
- Фильтрация дефектных панелей (размытые, слишком тёмные, только текст)

```
./work/panels/p001_001.jpg, p001_002.jpg, ...
```

### 4. Анализ панелей (Analyze)

LLM анализирует каждую панель через аннотированные изображения страниц (цветные рамки вокруг панелей — экономит до 85% токенов):
- Действие, персонажи, диалоги, звуковые эффекты
- Настроение, визуальные заметки
- Оценка важности (1-10)
- Определение дефектных панелей

### 5. Генерация сценария (Script)

Создаёт связный сценарий рекапа на основе анализа панелей:
- Фильтрация по минимальной важности
- Разбивка на сцены с индивидуальной озвучкой каждой панели
- Настроение, темп, тип переходов

```
./work/script.json
```

### 6. Озвучка (Voiceover)

Синтезирует речь через Qwen3-TTS:
- Отдельный аудиосегмент для каждой панели
- Кастомный голос и стиль
- Нормализация уровня громкости

```
./work/audio/panel_001_001.wav, ...
```

### 7. Рендеринг видео (Render)

Собирает финальное MP4 видео:
- Размытый фон + чёткая панель поверх
- Эффект Ken Burns (плавный зум/панорамирование)
- Синхронизация аудио с панелями
- Переходы между сценами

```
./work/output.mp4
```

---

## Примеры

### Базовый пример

```bash
# Минимальный запуск
recaper process ./One_Piece_ch1.cbz -o ./op_recap -t "One Piece Chapter 1"
```

<details>
<summary>Пример вывода</summary>

```
TODO: вставить вывод
```

</details>

### Пример с тонкой настройкой

```bash
# Высокое качество, только важные панели
RECAPER_LLM_JPEG_QUALITY=95 \
RECAPER_MIN_PANEL_IMPORTANCE=6 \
RECAPER_KEN_BURNS_ZOOM=1.08 \
recaper process ./manga.cbz -o ./work -t "Title" -v
```

### Пример работы с манхвой

```bash
recaper process ./tower_of_god/ -o ./tog_recap -t "Tower of God"
```

### Структура рабочей директории после обработки

```
work/
├── pages/              # Извлечённые страницы
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── panels/             # Вырезанные панели
│   ├── p001_001.jpg
│   ├── p001_002.jpg
│   └── ...
├── metadata.json       # Метаданные панелей
├── analysis/           # Результаты анализа
├── script.json         # Сценарий рекапа
├── audio/              # Аудиосегменты
│   ├── panel_001_001.wav
│   └── ...
└── output.mp4          # Финальное видео
```

---

## Структура проекта

```
recaper/
├── .env.example            # Шаблон конфигурации
├── pyproject.toml          # Зависимости и метаданные
├── run.py                  # Скрипт запуска веб-сервера
├── README.md               # README (English)
├── README.ru.md            # README (Русский)
├── LICENSE                 # GPL-3.0
├── src/recaper/
│   ├── __init__.py
│   ├── __main__.py         # Точка входа (python -m recaper)
│   ├── config.py           # Конфигурация (Pydantic Settings)
│   ├── exceptions.py       # Кастомные исключения
│   ├── models.py           # Модели данных (Pydantic)
│   ├── cli/
│   │   └── app.py          # CLI команды (Typer)
│   ├── pipeline/
│   │   ├── context.py      # Контекст пайплайна
│   │   ├── progress.py     # Прогресс-бар (Rich)
│   │   ├── runner.py       # Оркестратор пайплайна
│   │   └── stages/         # 7 этапов обработки
│   │       ├── unpack.py
│   │       ├── detect.py
│   │       ├── extract.py
│   │       ├── analyze.py
│   │       ├── script.py
│   │       ├── voiceover.py
│   │       └── render.py
│   └── web/
│       ├── app.py          # FastAPI приложение
│       ├── routes/         # API и страницы
│       ├── services/       # Сервис задач
│       ├── static/         # CSS/JS
│       └── templates/      # Jinja2 шаблоны
└── tests/
    ├── conftest.py
    └── test_pipeline/
```

---

## FAQ / Troubleshooting

### `recaper: command not found`

Убедитесь, что виртуальное окружение активировано и пакет установлен:
```bash
source .venv/bin/activate  # Linux
.venv\Scripts\activate     # Windows
pip install -e .
```

### `ffmpeg not found`

ffmpeg должен быть установлен и доступен в PATH:
```bash
ffmpeg -version
```

### `RECAPER_OPENROUTER_API_KEY is required`

Создайте файл `.env` на основе `.env.example` и добавьте ваш API ключ:
```bash
cp .env.example .env
# Отредактируйте .env
```

### Ошибки TTS / CUDA

TTS требует GPU с CUDA. Если GPU нет — этап озвучки будет пропущен или завершится ошибкой. Убедитесь что установлены CUDA-совместимые версии PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Пайплайн упал на середине

Используйте `--resume` чтобы продолжить с последнего завершённого этапа:
```bash
recaper process ./manga.cbz -o ./work --resume
```

### Слишком много токенов / дорогие запросы

- Уменьшите `RECAPER_LLM_JPEG_QUALITY` (например, `60`)
- Используйте более дешёвую модель: `RECAPER_OPENROUTER_MODEL=google/gemini-2.0-flash-001`
- Увеличьте `RECAPER_LLM_BATCH_SIZE` для меньшего числа запросов

### Плохое качество детекции панелей

- Попробуйте снизить `RECAPER_PANEL_CONFIDENCE` (например, `0.3`)
- Уменьшите `RECAPER_MIN_PANEL_AREA_RATIO` для мелких панелей

---

## Лицензия

[GPL-3.0-or-later](LICENSE)
