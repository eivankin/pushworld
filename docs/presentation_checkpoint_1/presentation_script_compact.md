# Компактный скрипт выступления

Целевая длина: около 2--3 минут.

## Слайд 1: Титульный

`На этом чекпоинте мы смотрим на PushWorld прежде всего как на задачу оптимизации AI-пайплайна обучения, а не как на соревнование по финальному success rate.`

## Слайд 2: Что такое PushWorld

`PushWorld --- это planning benchmark: агент двигается по сетке, толкает объекты, а на сложных уровнях появляются инструменты, узкие проходы и длинные причинно-следственные цепочки.`

`Поэтому нас интересует не только качество политики, но и то, где ломается training pipeline по времени.`

## Слайд 3: Как выглядит решение в среде

`Слева DQN на compact observations решает простую train-карту. Справа PPO меняет состояния, но не находит полезную последовательность действий и упирается в лимит шагов.`

`Это показывает, что бейзлайны отличаются и по качеству, и по поведению.`

## Слайд 4: Бейзлайны слабые и медленные

`PPO на RGB оказался нестабильным даже на пяти простых головоломках. DQN был устойчивее, но очень медленным.`

`Это были короткие прогоны порядка 100--200 тысяч итераций, так что по accuracy это ещё не финальный вердикт. Но уже видно, что базовая конфигурация слабая, а для нас сейчас важнее FPS и bottleneck analysis.`

## Слайд 5: RGB против planes

`Самая сильная ранняя оптимизация --- переход с RGB на planes. RGB тратит много времени на рендер и большие тензоры, а planes оставляют только полезную структуру состояния.`

`Это дало примерно `3x` для PPO и около `20x` для DQN.`

## Слайд 6: Bottleneck

`После planes среда уже не главный bottleneck. Для PPO и DQN основное время уходит в learner/update stack, а не в симуляцию.`

`Отсюда вывод: GPU-реализация среды пока не следующая оптимизация.`

## Слайд 7: torch.compile

`Мы также проверили model-side оптимизацию через torch.compile. Inference ускорился примерно в `1.4--1.6x`, но полный training step --- только на `3--4%`.`

`То есть compile полезен, но он не решает доминирующий bottleneck сам по себе.`

## Слайд 8: Пайплайн, который мы ускоряем

`Фиксируем базовый pipeline: state -> observation -> policy model -> rollout/replay -> learner update. Вход уже ускорен через planes; дальше для него идут goal-conditioned planes + relabeling, update profiling, AMP, rollout length, minibatch size, update epochs и replay/buffer layout.`

## Слайд 9: Transformer policy

`Transformer pipeline: planner solutions -> state-action dataset -> plane encoder + transformer -> action/distance heads -> greedy or beam search. Оптимизации: cached dataset, AMP/compile, larger batches, batched beam scoring, state/logit cache.`

## Слайд 10: Transformer optimization

`План: экспортировать planner traces, кэшировать plane tensors, сравнить CNN-only baseline с transformer over last k states и distance head, затем оптимизировать dataloader, AMP/compile, batch size, beam width и batched candidate scoring.`

## Слайд 11: Hybrid solver

`Hybrid pipeline: current state -> planner subgoals -> goal-conditioned executor -> progress monitor -> accept or replan. Цель --- заменить full search на planner skeleton + fast learned local execution.`

## Слайд 12: Hybrid optimization

`Оптимизации: offline subgoal traces, executor imitation/relabeling, profile planner/executor/env/replan time, cache subgoal plans, replan only on failure, learned ranker/value for planner candidates.`

## Слайд 13: RAGEN pipeline

`RAGEN branch: PushWorld Gym wrapper, deterministic text renderer, strict action parser, reward/success adapter. Baseline 1: fixed prompts, no fine-tune, 256 eval episodes. Baseline 2: LoRA/StarPO на маленьком Level 0 split.`

## Слайд 14: RAGEN optimization

`Оптимизации: measure wall time, tokens/sec, tokens/episode, generation/update/env-step split, GPU memory; then compact prompt, reasoning-token cap, action-only answer, vLLM rollout batching, Ray/env placement, LoRA vs full update, high-signal rollout filtering.`

## Слайд 15: Источники

`Тут опоры для следующих шагов: HER для relabeling, Tim Wheeler для Sokoban/transformer policy, Learn to Follow для hybrid planner + learning и RAGEN для LLM-agent RL.`

## Слайд 16: Код и материалы

`Здесь QR-код на репозиторий и прямая ссылка на GitHub, чтобы после доклада можно было быстро открыть проект.`

## Короткий ответ на вопрос про ценность текущих результатов

`Даже если финальная архитектура будет другой, compact observations, profiling и ускоренная инфраструктура обучения всё равно переиспользуются дальше. Поэтому текущие результаты полезны не только для PPO, но и для relabeling, transformer models и hybrid solver.`
