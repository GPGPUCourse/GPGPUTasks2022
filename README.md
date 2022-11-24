В этом репозитории предложены задания для курса по вычислениям на видеокартах 2022

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2022/).

# Задание 5. Bitonic sort, prefix sum, radix sort

[![Build Status](https://github.com/GPGPUCourse/GPGPUTasks2022/actions/workflows/cmake.yml/badge.svg?branch=task05&event=push)](https://github.com/GPGPUCourse/GPGPUTasks2022/actions/workflows/cmake.yml)

0. Сделать fork проекта
1. Выполнить задания 5.1, 5.2, 5.3
2. Отправить **Pull-request** с названием ```Task05 <Имя> <Фамилия> <Аффиляция>``` (указав вывод каждой программы при исполнении на вашем компьютере - в тройных кавычках для сохранения форматирования)

**Дедлайн**: начало лекции 1 ноября.

Задание 5.1. Bitonic sort
=========

Реализуйте bitonic sort для вещественных чисел (используя локальную память, пока размер подмассива для сортировки не станет слишком большим).

Файлы: ```src/main_bitonic.cpp``` и ```src/cl/bitonic.cl```

Задание 5.2. Prefix sum
=========

Реализуйте prefix sum в модели массового параллелизма.

Файлы: ```src/main_prefix_sum.cpp``` и ```src/cl/prefix_sum.cl```

Задание 5.3. Radix sort
=========

Реализуйте radix sort для unsigned int (используя локальную память).

Файлы: ```src/main_radix.cpp``` и ```src/cl/radix.cl```
