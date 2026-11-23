# Система распознавания лиц в реальном времени на Raspberry Pi

Проект реализует распознавание лиц в реальном времени с использованием OpenCV, ONNX Runtime и модели InsightFace.

Программа получает изображения с камеры, извлекает эмбеддинги лиц с помощью ONNX-модели, сравнивает их с заранее подготовленной базой и отображает результат распознавания.

## Возможности

- Захват изображения с USB-камеры или камеры Raspberry Pi.
- Обнаружение лиц с помощью Haar Cascade.
- Извлечение эмбеддингов с помощью ONNX Runtime.
- Сравнение лиц с заранее подготовленной базой пользователей.
- Отображение имени и уровня уверенности.
- Отображение рамок вокруг обнаруженных лиц.
- Поддержка нескольких пользователей.
- Отслеживание обнаруженных лиц и плавное исчезновение рамок.

## Структура проекта

```text
RPI-Face-Recognition/
├── Dataset/
│   ├── person_1/
│   └── person_2/
├── models/
│   └── buffalo_m/
│       └── w600k_r50.onnx
├── recognizer.py
├── recognizer_pi.py
├── requirements.txt
├── requirements_rpi.txt
└── README.md
```

`Dataset/` и `models/` содержат пользовательские данные и модели. Они не включаются в репозиторий автоматически.

## Требования

### Аппаратное обеспечение

- Raspberry Pi 3B+, Raspberry Pi 4 или Raspberry Pi 5.
- Камера Raspberry Pi (например, Camera Module v2 с сенсором IMX219) либо USB-камера.
- Дисплей и графическая среда для отображения окна распознавания.

### Программное обеспечение

- Raspberry Pi OS (Bookworm или Trixie).
- Python 3.10 или новее.
- Для Raspberry Pi Camera — Picamera2 и libcamera.
- OpenCV 4.x.
- ONNX Runtime.

Для Raspberry Pi OS рекомендуется использовать системные пакеты для доступа к камере и виртуальное окружение Python с параметром `--system-site-packages`.

## Установка на Raspberry Pi

### 1. Обновление системы

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Установка системных зависимостей

```bash
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    python3-libcamera
```

Picamera2 и libcamera должны устанавливаться через APT. Не устанавливайте их повторно через pip.

Для работы с Raspberry Pi Camera проверьте доступность камеры:

```bash
rpicam-hello
```

Если команда недоступна, проверьте установку пакетов камеры и конфигурацию системы.

### 3. Создание виртуального окружения

Перейдите в каталог проекта:

```bash
cd RPI-Face-Recognition
```

Создайте виртуальное окружение с доступом к системным Python-пакетам:

```bash
python3 -m venv --system-site-packages .venv
```

Активируйте его:

```bash
source .venv/bin/activate
```

Обновите pip:

```bash
python -m pip install --upgrade pip
```

**Важно:** параметр `--system-site-packages` необходим для того, чтобы виртуальное окружение могло использовать системные модули `picamera2` и `libcamera`.

### 4. Установка Python-зависимостей

Для Raspberry Pi:

```bash
python -m pip install -r requirements_rpi.txt
```

Для обычного компьютера:

```bash
python -m pip install -r requirements.txt
```

В проекте используется стандартный пакет `opencv-python`, а не `opencv-python-headless`, поскольку программа отображает изображение через `cv2.imshow()`.

Не устанавливайте одновременно несколько вариантов OpenCV:

- `opencv-python`
- `opencv-contrib-python`
- `opencv-python-headless`
- `opencv-contrib-python-headless`

Все эти пакеты используют один модуль Python — `cv2`.

Для Raspberry Pi не следует устанавливать OpenCV одновременно через APT и pip в одном окружении, если требуется использовать pip-сборку с `cv2.data.haarcascades`.

### 5. Проверка установки

В активированном виртуальном окружении выполните:

```bash
python - <<'PY'
import cv2
import numpy
import onnxruntime

print("OpenCV:", cv2.__version__)
print("OpenCV location:", cv2.__file__)
print("NumPy:", numpy.__version__)
print("ONNX Runtime:", onnxruntime.__version__)

print("CascadeClassifier:", hasattr(cv2, "CascadeClassifier"))
print("Haar cascades:", cv2.data.haarcascades)

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Cascade loaded:", not cascade.empty())
PY
```

Для Raspberry Pi Camera дополнительно проверьте:

```bash
python -c "from picamera2 import Picamera2; print('Picamera2: OK')"
```

Ожидаемый результат — успешный импорт модулей и `Cascade loaded: True`.

## Установка ONNX-модели

Проект использует модель InsightFace в формате ONNX.

Рекомендуемая модель:

- `buffalo_m/w600k_r50.onnx` — баланс производительности и точности.
- `buffalo_s/w600k_mbf.onnx` — более лёгкая модель.
- `antelopev2/glintr100.onnx` — более тяжёлая модель.

Модели можно получить из официального репозитория InsightFace:

https://github.com/deepinsight/insightface/releases/tag/v0.7

Для используемой по умолчанию модели создайте каталог:

```bash
mkdir -p models/buffalo_m
```

Поместите файл `w600k_r50.onnx` в:

```text
models/buffalo_m/w600k_r50.onnx
```

Путь к модели задаётся в исходном коде через `MODEL_PATH`.

## Подготовка датасета

В каталоге `Dataset/` необходимо создать отдельную папку для каждого человека.

Пример:

```text
Dataset/
├── andrey/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── max/
│   ├── photo1.jpg
│   └── photo2.jpg
└── anna/
    ├── photo1.jpg
    └── photo2.jpg
```

Требования:

- Название папки используется как имя человека.
- Поддерживаются изображения `.jpg`, `.jpeg` и `.png`.
- Рекомендуется использовать не менее пяти фотографий каждого человека.
- Желательно использовать фотографии с разными углами поворота головы и освещением.
- Изображения должны содержать лицо достаточно крупного размера.

При запуске программа загружает изображения, вычисляет эмбеддинги и формирует базу известных лиц.

## Запуск

Перед запуском активируйте виртуальное окружение:

```bash
source .venv/bin/activate
```

### Raspberry Pi Camera

Для камеры Raspberry Pi используйте:

```bash
python recognizer_pi.py
```

Этот вариант использует Picamera2 для захвата кадров.

### USB-камера

Для обычной USB-камеры используйте:

```bash
python recognizer.py
```

Этот вариант использует `cv2.VideoCapture()`.

Для выхода из окна распознавания нажмите `q`.

## Настройка распознавания

Основные параметры находятся в начале исходных файлов:

- `MODEL_PATH` — путь к ONNX-модели.
- `DATASET_DIR` — каталог с фотографиями пользователей.
- `CONFIDENCE_THRESHOLD` — минимальный порог уверенности распознавания.
- `FRAME_SKIP` — интервал обработки кадров.
- `FADE_TIMEOUT_SEC` — время до начала исчезновения рамки.
- `FADE_SPEED` — скорость исчезновения рамки.

Изменяйте эти параметры в соответствии с производительностью устройства и условиями съёмки.

## Устранение неполадок

### `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`

Проверьте, какая сборка OpenCV загружается:

```bash
python -c "import cv2; print(cv2.__version__, cv2.__file__)"
```

Проверьте наличие нескольких конфликтующих пакетов:

```bash
python -m pip list | grep -i opencv
```

В одном виртуальном окружении должен использоваться только один вариант OpenCV.

Для этого проекта рекомендуется `opencv-python==4.12.0.88`.

### `AttributeError: module 'cv2' has no attribute 'data'`

Вероятнее всего, загружена системная сборка OpenCV вместо pip-сборки, используемой проектом.

Проверьте:

```bash
python -c "import cv2; print(cv2.__file__); print(hasattr(cv2, 'data'))"
```

Убедитесь, что виртуальное окружение использует установленный через pip пакет OpenCV.

### `ModuleNotFoundError: No module named 'picamera2'`

Установите системный пакет:

```bash
sudo apt install python3-picamera2 python3-libcamera
```

Убедитесь, что виртуальное окружение создано с параметром:

```bash
python3 -m venv --system-site-packages .venv
```

### Ошибка `The function is not implemented` при вызове `cv2.imshow()`

Проверьте, что установлен пакет `opencv-python`, а не `opencv-python-headless`.

Также убедитесь, что программа запущена в графической сессии с доступным дисплеем.

### ONNX-модель не найдена

Проверьте наличие файла модели и правильность `MODEL_PATH`.

### Нет известных лиц

Проверьте структуру каталога `Dataset/` и наличие поддерживаемых изображений внутри папок пользователей.

## Примечания

- Не добавляйте виртуальное окружение `.venv/` в Git.
- Не добавляйте пользовательские фотографии и ONNX-модели в публичный репозиторий без необходимости.
- Для Raspberry Pi устанавливайте системные компоненты камеры через APT.
- Не смешивайте несколько сборок OpenCV в одном Python-окружении.
- Для корректной работы графического интерфейса используйте обычную сборку `opencv-python`.

## Лицензия

См. файл LICENSE в репозитории, если он присутствует.