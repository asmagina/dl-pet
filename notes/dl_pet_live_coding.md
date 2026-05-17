#flashcards/dl-pet

Лайв-кодинг на собесе ≈ **один `train.py`**: константы/argparse сверху, 2–3 функции, `if __name__`. Пакет `dl_pet/` — для своего репо, не для доски.

**Ментальная модель:** Config (строки вверху) → DataLoader → Model → `train_epoch` / `eval_epoch` → цикл `for epoch`.

---

## Один файл: что на доске

Минимальный `train.py` — что объявить сверху вниз (6 блоков):::
imports
константы или argparse (DATA_ROOT, BATCH, LR, EPOCHS, DEVICE)
Dataset/transforms + DataLoader (train, val)
class Model(nn.Module)
def train_epoch(...)
def eval_epoch(...)
if __name__ == "__main__": main()

Сколько файлов на лайв-кодинге в 90% случаев?:::
1 (`train.py`); иногда 2 если модель большая (`model.py` + `train.py`)

Когда на собесе всё-таки дробят?:::
после того как скрипт работает; или если интервьюер дал заготовку с пакетом; не с нуля ради архитектуры

Наш `dl_pet/` с yaml — это что?:::
прод/пет-проект; на доске то же самое, но сжато в один файл

---

## Константы / argparse (вместо yaml)

Типичные константы вверху `train.py` (6):::
DATA_ROOT, BATCH_SIZE, NUM_EPOCHS, LR, DEVICE, maybe IMAGE_SIZE

Строка DEVICE одной строкой:::
`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

argparse: что спросят чаще констант?:::
`--data-root`, `--epochs`, `--lr`, `--batch-size` — опционально, константы быстрее на 45 мин

Забыл вынести lr/epochs вверх — не страшно?:::
да; главное — правильный train/eval loop, не структура папок

---

## Data (в том же файле)

ImageFolder + DataLoader — 4 строки логики:::
transforms = Compose([Resize, ToTensor, Normalize(...)])
train_ds = ImageFolder(train_root, transform=transforms)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, pin_memory=True)

Train loader vs val loader — два отличия:::
shuffle=True только train; val shuffle=False

Аугментация только на train — пример одной строки:::
`RandomHorizontalFlip()` в Compose train, не в val

`pin_memory=True` — зачом на собесе одной фразой?:::
быстрее копировать батч на GPU с CPU

`num_workers` на доске:::
часто `0` или `2` — чтобы не утонуть в multiprocessing на VM

---

## Model (в том же файле)

Минимальный `nn.Module` — что обязательно:::
`__init__` слои, `forward(self, x)` return logits (без softmax под CrossEntropy)

После conv блоков перед linear:::
`x.flatten(1)` или `AdaptiveAvgPool2d` + flatten

`model.to(device)` — где вызвать?:::
один раз после создания, до цикла epoch

---

## train_epoch — это главное

Сигнатура на доске:::
`def train_epoch(model, loader, criterion, optimizer, device):`

Первая строка тела:::
`model.train()`

Один батч — порядок строк (7):::
images, targets = images.to(device), targets.to(device)
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()

Что НЕ писать в train_epoch:::
`torch.no_grad()`, `model.eval()`

CrossEntropyLoss — что подаёшь в model?:::
logits (сырой выход), не softmax

---

## eval_epoch

Сигнатура:::
`def eval_epoch(model, loader, criterion, device):`

Две строки до цикла for:::
`model.eval()`
весь цикл в `with torch.no_grad():` (или `@torch.no_grad()` на функции)

Один батч eval — чего нет (3):::
zero_grad, backward, optimizer.step

Как посчитать accuracy за эпоху без класса AverageMeter:::
сумма correct, сумма total по батчам; return correct/total

---

## main() внизу файла

Порядок в `main()` (7 шагов):::
device
loaders
model, criterion, optimizer
loop epochs: train_epoch → eval_epoch → print
опционально torch.save state_dict лучшей модели

Optimizer одной строкой:::
`optimizer = torch.optim.Adam(model.parameters(), lr=LR)`

Criterion для classification:::
`nn.CrossEntropyLoss()`

Сохранение «лучшей» модели — минимум:::
```python
if val_acc > best:
    best = val_acc
    torch.save(model.state_dict(), "best.pt")
```

Загрузка для дообучения / eval-only:::
`model.load_state_dict(torch.load("best.pt", map_location=device))`

---

## Cloze: следующая строка

#flashcards/dl-pet/cloze

После `loss = criterion(logits, targets)`:::
`loss.backward()`

После `loss.backward()`:::
`optimizer.step()`

Перед `logits = model(images)`:::
`optimizer.zero_grad()`

Перед циклом по val loader:::
`model.eval()`

Внутри eval на батче, перед forward:::
уже внутри `torch.no_grad()` контекста

---

## Напиши с нуля (один файл)

#flashcards/dl-pet/write

Весь `train_epoch` без метрик:::
```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
```

Весь `eval_epoch` с return acc:::
```python
def eval_epoch(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            preds = logits.argmax(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total
```

Скелет `main` — только цикл:::
```python
for epoch in range(NUM_EPOCHS):
    train_epoch(model, train_loader, criterion, optimizer, device)
    acc = eval_epoch(model, val_loader, criterion, device)
    print(epoch, acc)
```

Минимальный `DataLoader` блок:::
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
train_loader = DataLoader(
    ImageFolder("data/train", transform=transform),
    batch_size=32,
    shuffle=True,
)
```

---

## Когда спросят «как в проекте» (не пиши с нуля)

Интервьюер: «вынеси в функции/модули» — минимальное дробление (3 части):::
`train.py` — main + epoch loops
`data.py` — get_loaders()  (опционально)
`model.py` — class Net

Зачем в репо отдельный yaml, если на собесе не надо?:::
эксперименты без правок кода; в лайве — константы/argparse достаточно

`train_one_epoch` в пакете = что на доске?:::
та же `train_epoch`, просто другое имя

---

## Промахи (SW → DL)

#flashcards/dl-pet/mistakes

Забыл `model.train()` / `eval()`:::
Dropout/BatchNorm врут, val метрики фейк

Забыл `zero_grad`:::
градиенты копятся между шагами

CE + softmax в forward:::
лишнее; CE сама нормализует

targets не на device:::
ошибка в criterion

`shuffle=False` на train:::
хуже сходимость, не баг компиляции

---

## Reverse (кусок кода → где в одном файле)

#flashcards/dl-pet/reverse

`optimizer.zero_grad()` внутри for по loader:::
функция `train_epoch`

`with torch.no_grad()`:::
функция `eval_epoch`

`ImageFolder(..., transform=...)`:::
блок data в `main` или над ним

`torch.save(model.state_dict(), path)`:::
цикл epoch в `main` после eval

`argparse.ArgumentParser()`:::
верх файла; на доске часто заменяют константами
