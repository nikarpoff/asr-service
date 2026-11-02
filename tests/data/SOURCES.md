# Источники тестовых данных

Игрушечный манифест вопросов `questions.jsonl` в формате EXP-1 и записи к нему. Записи - произношения слов из The Shtooka Project, опубликованные на Wikimedia Commons. Каждая переведена в WAV 16 кГц моно 16 бит (`ffmpeg -ac 1 -ar 16000 -sample_fmt s16`).

| Файл | Слово | Условие | Источник | Автор | Лицензия |
|---|---|---|---|---|---|
| `questions/q001.wav` | дверь | `quiet` | [File:Ru-дверь.ogg](https://commons.wikimedia.org/wiki/File:Ru-%D0%B4%D0%B2%D0%B5%D1%80%D1%8C.ogg) | Сергей Сахно, Vion Nicolas, The Shtooka Project | [CC BY 3.0 US](https://creativecommons.org/licenses/by/3.0/us/) |
| `questions/q002.wav` | стул | `quiet` | [File:Ru-стул.ogg](https://commons.wikimedia.org/wiki/File:Ru-%D1%81%D1%82%D1%83%D0%BB.ogg) | The Shtooka Project | [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/) |
| `questions/q003.wav` | где | `street_noise` | [File:Ru-где.ogg](https://commons.wikimedia.org/wiki/File:Ru-%D0%B3%D0%B4%D0%B5.ogg) | The Shtooka Project | [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/) |
| `questions/q004.wav` | окно | `street_noise` | [File:Ru-окно.ogg](https://commons.wikimedia.org/wiki/File:Ru-%D0%BE%D0%BA%D0%BD%D0%BE.ogg) | The Shtooka Project | [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/) |

Изменения: в `q003.wav` и `q004.wav` к речи добавлен белый шум с SNR 10 дБ (генератор `numpy.random.default_rng(2026)`), он изображает условие `street_noise`. Шум синтетический.
