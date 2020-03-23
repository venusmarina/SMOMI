В данной задаче используется пример полносвязной нейронной сети, 
включающей в себя 5 слоёв. Первый слой - Flatten, ещё четыре - Dense 
(256, 128, 64 и 10 нейронов). Три из четырёх слоёв Dense используют 
активацию relu, а четвёртый - sigmoid.

В обучении нейронной сети производится 10 эпох обучения. 
В цикле обучения используются:

- оптимизатор SGB
- функция потерь sparse_categorical_crossentropy
- метрика accuracy

Функции потерь 

![Image alt](https://github.com/venusmarina/SMOMI/blob/master/Loss.png)

Метрики точности

![Image alt](https://github.com/venusmarina/SMOMI/blob/master/Accuracy.png)

Итоговая точность на тестовых данных 0.4913