# Репозиторий для классификации изображений

## Требования для запуска

1. Python 3.8 или выше.
2. Для GPU видеокарта с поддержкой CUDA 11.1.
3. [Доступ в Google Drive для DVC.](https://dvc.org/doc/user-guide/setup-google-drive-remote)

## Как запустить

### Зависимости

Установить зависимости:
```
pip install -r. /requirements.txt -r ./requirements.dev.txt
```

Скачать данные из DVC:
```
dvc pull
```

Установка PyTorch с GPU:
```
pip install -r ./requirements.gpu.txt
```

PyTorch с CPU только:
```
pip install -r ./requirements.cpu.txt
```

### Запуск обучения
```
dvc repro train
```

### Запуск тестов:
```
pytest --cov=model --cov=training ./tests
```
