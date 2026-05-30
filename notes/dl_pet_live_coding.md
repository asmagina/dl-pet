#flashcards/dl-pet

Текущая структура проекта минимальная и близкая к лайв-кодингу: без yaml, без argparse, без `loops.py`.

---

## Структура Текущего Кода

Какие основные файлы в нужны по-минимуму?:::
`config.py` — константы  
`dataloader.py` — transforms, Dataset, DataLoader  
`model.py` — CNN  
`train.py` — train/eval loop + обучение  
`test.py` — загрузка checkpoint + eval на test

Главная ментальная цепочка проекта:::
`config` → `dataloader` → `model` → `train_epoch` / `eval_epoch` → checkpoint → `test.py`

Что важно не забыть, если переименовываешь класс модели?:::
Имя класса в `model.py` должно совпадать с импортами в `train.py` и `test.py`.

---

## config.py

Что лежит в `config.py`?:::
пути, категория датасета, размер картинки, batch size, workers, epochs, lr, weight decay, seed, checkpoint paths, test range

Константы данных в `config.py`:::
`DATA_ROOT`, `CATEGORY`, `IMAGE_SIZE`, `BATCH_SIZE`, `NUM_WORKERS`

Константы обучения в `config.py`:::
`NUM_EPOCHS`, `LR`, `WEIGHT_DECAY`, `SEED`

Константы checkpoint в `config.py`:::
`CHECKPOINT_DIR`, `BEST_CKPT`

```

---

## dataloader.py:  Transforms

Какие две функции transforms есть сейчас?:::
`train_transform(image_size)` и `eval_transform(image_size)`.

Что есть в `train_transform`?:::
`Resize`, `HorizontalFlip`, `RandomBrightnessContrast`, `Rotate`, `Normalize`, `ToTensorV2`.

Что есть в `eval_transform`?:::
`Resize`, `Normalize`, `ToTensorV2`.

Почему train/eval transforms разные?:::
Аугментации нужны только на train; eval должен быть стабильным и детерминированным.

Что делает `A.Normalize()` без аргументов?:::
Использует дефолтные ImageNet mean/std в Albumentations.

Почему `ToTensorV2()` идёт после `A.Normalize()`?:::
Albumentations работает с numpy image; `ToTensorV2` в конце превращает результат в PyTorch tensor.

Horizontal flip в Albumentations — строка:::
`A.HorizontalFlip(p=0.5)`

Brightness/contrast в Albumentations — смысл:::
Слегка меняет освещение/контраст, чтобы модель не переобучалась на один вид картинки.

Rotate в train transform — зачем?:::
Небольшая устойчивость к повороту; в текущем коде `limit=15`, `p=0.3`.

---

## dataloader.py: Dataset

Зачем свой `ImageFolderAlbumentations`, если есть `ImageFolder`?:::
`ImageFolder` отдаёт PIL image, а Albumentations ожидает numpy image с ключом `image`.

Что хранит `ImageFolderAlbumentations.__init__`?:::
`self._folder = ImageFolder(root=root, transform=None)` и `self.transform = transform`.

Что делает `__len__` в dataset wrapper?:::
Возвращает `len(self._folder)`.

Порядок в `__getitem__`:::
получить `image, label` из `ImageFolder` → `np.array(image)` → `self.transform(image=image)` → вернуть `out["image"], label`

Почему transform вызывается как `self.transform(image=image)`?:::
Такой API у Albumentations: вход и выход — dict, картинка лежит по ключу `"image"`.

---

## dataloader.py: DataLoader

Сигнатура `build_dataloader` по смыслу:::
`root`, `image_size`, `batch_size`, `num_workers`, `train`, `shuffle`, optional `start_index`, `stop_index`

Как выбирается transform в `build_dataloader`?:::
`train_transform(image_size) if train else eval_transform(image_size)`

Зачем `shuffle=True` только на train?:::
Чтобы батчи менялись между эпохами; eval/test должен быть стабильным.

Зачем `Subset` в `build_dataloader`?:::
Чтобы прогнать только диапазон изображений, например `[100, 200)`.

Какой диапазон означает `start_index=10, stop_index=20`?:::
Индексы `[10, 20)`: 10 включительно, 20 не включительно.

Что делает код, если selected range пустой?:::
Бросает `ValueError("selected image range is empty")`.

`pin_memory=True` — коротко зачем?:::
Ускоряет перенос CPU batch на GPU; не критично для лайв-кодинга.

---

## model.py

Что должен содержать `model.py`?:::
Один класс модели: `class SimpleCNN(nn.Module)` с `__init__` и `forward`.

Минимальный контракт модели для `CrossEntropyLoss`:::
`forward(x)` возвращает logits формы `[batch_size, num_classes]`, без softmax.

Что делает `self.features` в модели?:::
Извлекает признаки через conv/relu/pool блоки.

Что делает `self.classifier`?:::
Линейный слой из feature vector в `num_classes`.

Зачем `AdaptiveAvgPool2d(1)`?:::
Сжимает пространственные размеры до `1x1`, чтобы classifier не зависел от точного размера feature map.

Что делает `x.flatten(1)`?:::
Оставляет batch dimension и расплющивает все остальные измерения.

Что должно совпадать между `model.py` и `train.py`?:::
Имя импортируемого класса: если `train.py` пишет `from model import SimpleCNN`, в `model.py` должен быть `SimpleCNN`.

---

## train.py: Helpers

Что делает `set_seed`?:::
Ставит seed для `random`, `numpy`, `torch`, `torch.cuda`.

Формула `accuracy(logits, targets)`:::
`preds = logits.argmax(dim=1)` → сравнить с `targets` → `correct / batch_size`.

Почему `argmax(dim=1)`?:::
`dim=1` — измерение классов в logits `[batch, classes]`.

---

## train.py: train_epoch

Сигнатура `train_epoch`:::
`model, loader, criterion, optimizer, device -> dict[str, float]`

Первая строка в `train_epoch`:::
`model.train()`

Порядок одного train batch:::
`to(device)` → `zero_grad()` → `model(images)` → `criterion` → `backward()` → `optimizer.step()`

Что обязательно перед `loss.backward()`?:::
Посчитать `loss = criterion(logits, targets)`.

Что обязательно перед `logits = model(images)`?:::
Перенести `images` и `targets` на `device`, затем `optimizer.zero_grad()`.

Почему `loss_sum += loss.item() * batch_n`, а не просто `loss_sum += loss.item()`?:::
Чтобы средний loss был взвешен по числу примеров, особенно если последний batch меньше.

Что возвращает `train_epoch`?:::
`{"loss": loss_sum / n, "acc": acc_sum / n}`

---

## train.py: eval_epoch

Декоратор `eval_epoch`:::
`@torch.no_grad()`

Первая строка в `eval_epoch`:::
`model.eval()`

Чего нет в eval batch?:::
`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.

Почему `@torch.no_grad()`?:::
Не строить граф градиентов: быстрее и меньше памяти.

Почему `model.eval()`?:::
Dropout/BatchNorm переходят в inference mode.

Что возвращает `eval_epoch`?:::
Такой же dict метрик: `{"loss": ..., "acc": ...}`.

---

## train.py: main

Порядок `main()` в `train.py`:::
seed → device → paths → train/val loaders → model/loss/optimizer → checkpoint dir → loop epochs

Строка выбора device:::
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`

Откуда берутся train и val roots?:::
`mvtec_split_root(config.DATA_ROOT, config.CATEGORY, "train")` и `"test"`.

Как создаётся модель в `train.py`?:::
`num_classes = len(train_loader.dataset.classes)` → `SimpleCNN(in_channels=3, num_classes=num_classes).to(device)`

Criterion сейчас:::
`nn.CrossEntropyLoss()`

Optimizer сейчас:::
`torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)`

Что происходит в epoch loop?:::
`train_epoch` → `eval_epoch` → print metrics → save `last.pt` → если лучше, save `best.pt`.

Что лежит в checkpoint dict?:::
`epoch`, `model`, `optimizer`, `classes`.

Когда сохраняется `best.pt`?:::
Когда `val_metrics["acc"] > best_acc`.

---

## test.py

Что делает `test.py`?:::
Загружает test DataLoader, checkpoint, модель, потом вызывает `eval_epoch`.

Почему `test.py` импортирует `eval_epoch` из `train.py`?:::
Чтобы не дублировать eval loop в маленьком проекте.

Какие config-поля использует `test.py` для диапазона картинок?:::
`TEST_START_INDEX`, `TEST_STOP_INDEX`.

Зачем `map_location=device` в `torch.load`?:::
Чтобы checkpoint загрузился на текущий CPU/GPU device.

Как определяется `num_classes` в `test.py`?:::
`num_classes = len(ckpt["classes"])`.

Что делает `model.load_state_dict(ckpt["model"])`?:::
Загружает веса модели из checkpoint.

Что печатает `test.py` перед eval?:::
`start`, `stop`, `count` выбранных test images.

---

## Cloze: Следующая Строка

#flashcards/dl-pet/cloze

После `loss = criterion(logits, targets)`:::
`loss.backward()`

После `loss.backward()`:::
`optimizer.step()`

Перед `logits = model(images)` в train batch:::
`optimizer.zero_grad()`

Перед циклом по train loader:::
`model.train()`

Перед циклом по eval loader:::
`model.eval()`

После `preds = logits.argmax(dim=1)` в accuracy:::
`correct = (preds == targets).sum().item()`

После `torch.load(config.BEST_CKPT, map_location=device, weights_only=False)` в test:::
создать модель с нужным `num_classes`, затем `model.load_state_dict(ckpt["model"])`

---

## Напиши С Нуля

#flashcards/dl-pet/write

Напиши минимальный `train_epoch`:::
```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
```

Напиши минимальный `eval_epoch`:::
```python
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return correct / total
```

Напиши минимальный `train_transform` на Albumentations:::
```python
def train_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])
```

Напиши минимальный `eval_transform` на Albumentations:::
```python
def eval_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2(),
    ])
```

Напиши минимальный checkpoint save:::
```python
torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "classes": train_loader.dataset.classes,
}, checkpoint_dir / "last.pt")
```

---

## Частые Ошибки

#flashcards/dl-pet/mistakes

Забыла `model.train()` / `model.eval()`:::
Dropout/BatchNorm работают не в том режиме; метрики могут быть неверными.

Забыла `optimizer.zero_grad()`:::
Градиенты копятся между batch steps.

Добавила softmax перед `CrossEntropyLoss`:::
Не надо: `CrossEntropyLoss` ждёт raw logits.

Забыла `targets.to(device)`:::
Будет device mismatch в loss.

Поставила `shuffle=False` на train:::
Обучение может стать хуже; train обычно shuffle.

Поставила аугментации в eval transform:::
Eval станет недетерминированным, метрики будут шуметь.

Переименовала `SimpleCNN`, но не поменяла import:::
`ImportError`: `train.py`/`test.py` не найдут класс.

---

## Reverse Cards

#flashcards/dl-pet/reverse

`A.Normalize()` без аргументов:::
Albumentations ImageNet normalization.

`Subset(dataset, indices)`:::
Ограничение test/train dataset диапазоном индексов.

`torch.no_grad()`:::
Eval/inference без графа градиентов.

`logits.argmax(dim=1)`:::
Получить predicted class index из logits.

`AdaptiveAvgPool2d(1)`:::
Сжать feature map до `1x1` перед linear classifier.

`checkpoint_dir / "best.pt"`:::
Файл лучшей модели по validation accuracy.

`map_location=device`:::
Загрузить checkpoint сразу на нужное устройство.
