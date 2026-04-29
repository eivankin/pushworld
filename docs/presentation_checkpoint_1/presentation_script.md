# Скрипт выступления

Целевая длина: около 5 минут.

## Слайд 1: Титульный

`На этом чекпоинте мы смотрим на PushWorld прежде всего как на задачу по оптимизации AI-пайплайна обучения, а не как на соревнование по финальному success rate.`

## Слайд 2: Что такое PushWorld

`PushWorld --- это planning benchmark: агент двигается по сетке, толкает объекты, а на более сложных картах появляются инструменты, узкие проходы и длинные причинно-следственные зависимости.`

`Для нас это удобный тест на то, где RL-пайплайн ломается и по времени, и по качеству.`

`Поэтому главный вопрос сегодня такой: куда уходит время в пайплайне обучения и какие оптимизации действительно имеют смысл делать следующими?`

## Слайд 3: Как выглядит решение в среде

`Здесь важно быстро показать саму динамику задачи. Слева DQN на compact observations решает простую train-карту. Справа PPO на том же типе наблюдений меняет состояния, но не находит полезную последовательность действий и упирается в лимит шагов.`

`То есть даже на маленьких задачах у нас уже видно, что разные алгоритмы ломаются по-разному, и это дополнительно мотивирует разбирать именно пайплайн, а не только финальные метрики.`

## Слайд 4: Бейзлайны слабые и медленные

`Стартовые бейзлайны были слабыми по-разному. PPO на RGB оказался нестабильным даже на пяти простых головоломках. DQN на том же наборе вёл себя устойчивее, но работал очень медленно по wall-clock времени.`

`Важно, что это пока короткие прогоны порядка 100--200 тысяч итераций. То есть по качеству это ещё не окончательный вердикт, но слабость бейзлайнов уже хорошо видна.`

`Для этого checkpoint главный сигнал здесь --- не итоговая accuracy, а системная сторона: какой FPS даёт конфигурация и куда вообще уходит время.`

## Слайд 5: Что видит агент: RGB против planes

`Первая high-leverage оптимизация была очень простой по идее, но сильной по эффекту: мы заменили RGB-наблюдения на компактные plane-наблюдения.`

`Это сработало потому, что RGB заставляет среду рендерить большую картинку и гонять большие тензоры через replay buffer и CNN. Plane-наблюдения оставляют только полезную структуру состояния.`

`В результате размер состояния резко уменьшился, PPO ускорился примерно втрое, а DQN --- примерно в двадцать раз.`

## Слайд 6: После planes среда уже не главный bottleneck

`Профилировка объясняет, почему это сработало. Для RGB большая часть измеренной стоимости среды была не в transition logic PushWorld, а в observation rendering. После перехода на planes стоимость среды стала занимать очень маленькую долю train time.`

`Потом мы векторизовали rollout для PPO. Это ещё ускорило обучение, но одновременно сдвинуло bottleneck: при шестнадцати окружениях PPO уже тратит около 77 процентов времени на update.`

`То же самое мы увидели и для DQN: после удаления RGB overhead среда занимает только около 0.38 процента времени, а update --- около 76 процентов.`

`Итог здесь простой: текущий bottleneck --- это learner/update stack, а не симуляция среды.`

## Слайд 7: Ещё одна оптимизация --- torch.compile

`Чтобы проверить уже не только representation-level, но и model-side оптимизацию, мы добавили CUDA microbenchmark для нашей PushWorld CNN и протестировали torch.compile.`

`Результат получился показательный: isolated inference ускорился примерно в 1.4--1.6 раза, то есть compile действительно ускоряет forward path. Но полный training step ускорился только на 3--4 процента.`

`Это значит, что compile --- реальная оптимизация, но она не бьёт по доминирующему bottleneck сама по себе. И это хорошо согласуется с профилировкой: проблема шире, чем просто скорость forward pass.`

## Слайд 8: Пайплайн, который мы ускоряем

`Теперь явно фиксируем базовый pipeline: PushWorld state, plane observation, policy model, rollout или replay, learner update.`

`Вход уже ускорен: RGB rendering заменён на compact planes, observation payload стал примерно в 50 раз меньше, PPO ускорился примерно втрое, DQN --- примерно в двадцать раз.`

`Дальше для этого же pipeline есть понятные шаги: goal-conditioned planes и relabeling, update profiling, AMP, настройка rollout length, minibatch size, update epochs и replay/buffer layout. Основная метрика --- success/time и update seconds per 100k env steps.`

## Слайд 9: Transformer policy pipeline

`Первый advanced pipeline --- transformer policy. Planner даёт solution traces, мы превращаем их в state-action dataset, обучаем plane encoder плюс transformer, а на выходе получаем action head и distance или solvability head.`

`На inference проверяются greedy rollout и beam search. Здесь оптимизируется supervised training и autoregressive search вокруг policy, а не environment stepping. Основа --- работы Wheeler по Sokoban transformer policy и GPU rollout.`

## Слайд 10: Transformer policy: план оптимизации

`Сначала экспортируем planner traces и кэшируем plane tensors, чтобы training не ждал парсер или симулятор. Потом делаем model ablation: CNN-only no-history baseline, transformer по последним k states, затем distance head.`

`После этого оптимизируем training: dataloader, AMP, torch.compile, larger batches. На search side: greedy, beam width 8/16/32, batched scoring всех beam candidates одним forward, state/logit cache.`

## Слайд 11: Hybrid solver pipeline

`Второй advanced pipeline --- hybrid solver. Planner задаёт subgoals, learned executor локально исполняет primitive actions, monitor проверяет прогресс, и replanning запускается только при timeout или регрессе относительно subgoal.`

`Цель --- заменить дорогой full search на planner skeleton плюс fast learned local execution. Метрики: subgoal success, replans per puzzle, planner time versus executor time, solved puzzles per minute.`

## Слайд 12: Hybrid solver: план оптимизации

`Сначала берём solution traces и нарезаем их на достижимые partial goals через каждые k действий. Executor baseline учится на этих subgoals через imitation или relabeling.`

`Затем отдельно профилируем planner call time, executor inference, env stepping и replanning overhead. Оптимизации: cache subgoal plans, replan only on failure, learned ranker/value вместо полного rollout search по нескольким candidates. Ablation: planner-only, executor-only, hybrid without ranker, hybrid with ranker.`

## Слайд 13: RAGEN pipeline: LLM-agent RL

`Третий advanced pipeline --- RAGEN, более новая LLM-agent RL ветка. RAGEN обучает LLM agents через multi-turn environment interaction: state text, reasoning/action, reward, затем policy update.`

`PushWorld adaptation здесь конкретная: Gym wrapper, deterministic text renderer, strict action parser, reward/success adapter. Baselines: fixed prompt without fine-tune на 256 episodes; затем LoRA/StarPO на маленьком Level 0 split.`

## Слайд 14: RAGEN pipeline: план оптимизации

`RAGEN оптимизируем как тяжёлый LLM/RL pipeline. Сначала baseline profiling: wall time, tokens/sec, tokens per episode, env-step time, generation time, update time, GPU memory.`

`Затем interventions: prompt compression, reasoning-token cap, one-token action answer, rollout batching через vLLM, Ray worker/env placement, no-finetune versus LoRA versus full update, high-signal rollout filtering. RAGEN paper даёт runtime context, но не полноценную throughput table.`

## Слайд 15: Источники и ориентиры

`Здесь собраны внешние опоры для следующих шагов: HER для relabeling, Tim Wheeler для transformer-policy направления, Learn to Follow для hybrid planner + learning подхода и RAGEN как ориентир по LLM-agent RL.`

`На этом можно закончить основную часть и перейти к вопросам.`

## Слайд 16: Код и материалы

`Здесь можно быстро показать QR-код на репозиторий и оставить прямую ссылку на GitHub, чтобы материал было проще открыть после доклада.`

## Ответы на вероятные вопросы

Если спросят, почему GPU-реализация среды не следующая:

`Потому что после перехода на planes измеренная стоимость среды очень мала и для PPO, и для DQN. Сейчас основное время уходит в learner/update stack, поэтому GPU simulator был бы преждевременной оптимизацией.`

Если спросят, почему текущие оптимизации полезны, если PPO всё равно слаб:

`Потому что relabeling, goal-conditioned policy, hybrid solver и offline transformer всё равно переиспользуют compact state, rollout/eval инфраструктуру и bottleneck analysis. То есть эти выводы прямо переносятся дальше.`

Если спросят, почему мы вообще смотрим на planning-style идеи:

`Потому что PushWorld по сути является planning benchmark, и текущие PPO/DQN служат нам скорее измерительным baseline, чем финальной архитектурой.`

Если спросят, нужно ли прямо сейчас адаптировать RAGEN:

`Да. План не в том, чтобы рассуждать, полезен ли RAGEN вообще, а в том, чтобы сделать adapter, получить prompt-only baseline, измерить bottleneck split и затем оптимизировать конкретные места: prompt length, reasoning budget, rollout batching, vLLM/Ray settings, LoRA и filtering.`
