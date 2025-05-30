


Результаты:

| Название агента/агентной системы | Датасет | Метрика | Значение метрики | Время выполнения | Затрачено входных токенов | Затрачено выходных токенов |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
|ML engineer | user churn | ROC-AUC | 0.8015264642442343 | 48.1s | 14528 | 1730 |
|MLE + planning | user churn | ROC-AUC | 0.7987058237929316    |  58s |  15545   |  1337 |
|MLE + Human Insight Bank (HIB)| user churn| ROC-AUC |0.8070225651236104|3m 8.6s|366 282|2 576|
|MLE + HIB + planning|user churn| ROC-AUC |0.7987|2m 58.0s|305 264|17 691|
|Manager + MLE|user churn| ROC-AUC |0.8215|29.8s|15297|5411|
|Manager + MLE + planning|user churn| ROC-AUC |0.8214|64s|33041|10270|
|MLE|flat price|RMSE|15100681.248412501|20m 4.3s|335 149|24 843|
|MLE + planning|flat price|RMSE|13747906.007218136|35m|35007|5691|
|MLE + HIB|flat price|RMSE|18563619.853423897|53.6s|38 827|3 261|
|MLE + HIB + planning|flat price|RMSE|15716963.956103794|3m 11.8s|131 649|10 043|
|Manager + MLE|flat price| RMSE |15504562.0264|41.6s|68236|8566|
|Manager + MLE + planning|flat price| RMSE |14423296.2840 |68.22s|24347|6445|
