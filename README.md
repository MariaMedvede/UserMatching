# UserMatching
It's a project for analyzing similarity between two profiles in a social site.

Для начала представлен алгоритм оценки схожести текстов. 

## Web scrapping 

Так как для скачивания описаний к фотографиям Инстаграм с 29 июня 
2019 года не предоставляет свой API, выкачивание данных производилось с 
помощью автоматизированного ПО. Для этих целей был использован пакет 
Selenium for Python, который превосходит другие библиотеки, например, как 
Beautiful Soup, своей способностью хорошо работать на сайтах на Javascript. 

Сначала, с помощью функции recent_posts мы устанавливаем 
соединение со страницей пользователя, получаем ссылки на посты и 
записываем их в список. Так как на странице отображаются все посты не 
сразу, а только при пролистывании вниз, после каждого выкачивания ссылок 
мы пролистываем страницу в конец, и ждем некоторое время, чтобы 
прогрузились новые ссылки. Если мы достигли конца страницы и никаких 
новых ссылок не обнаружено, либо мы уже собрали 150 постов, функция 
прекращает свое выполнение. На выходе из этой функции мы имеем список ссылок на посты пользователя.  

Дальше функция insta_details на входе принимает список из ссылок на 
посты, проходит по каждой ссылке и копирует описание к фото или видео. 
Так как мы не можем обратиться к описанию с помощью тега, потому что он 
используется не только для описания, мы находим на странице описание по 
xpath. На выходе у нас также получается список, уже из всех описаний к 
фотографиям или видео. В конце мы сохраняем это описание в json файл, так 
как каждый раз выполнять эту программу заново занимает очень много 
времени.  

## Preprocessing 

Чтобы привести текст в подходящий для анализа вид, необходимо 
сделать несколько шагов. Во-первых, мы избавились от цифр, пунктуации, 
смайлов и других ненужных знаков, оставив только буквы с помощью 
регулярных выражений.  
Во-вторых, с помощью библиотеки Mystem[1], единственной для 
русского языка, мы лемматизировали все слова. Из библиотеки nltk для 
обработки естественного языка мы нашли и удалили русские стоп-слова.  
В-третьих, необходимо было применить TF-IDF векторайзер на списке 
слов пользователей. Встроенный в sklearn векторайзер не подошел для наших 
входных данных, поэтому была написана своя функция, выполняющая TF-
IDF.  
Получив для каждого пользователя свой вектор, мы посчитали их 
похожесть с помощью функции cosine_similarity из sklearn. 

Оценка схожести фотографий происходит по алгоритму, изображенному на "1.png".

Общий алгоритм оценки схожести изображен на "2.png", где функция рекомендации

h(Лица(u, s), Текст(u,s)) = 0,4*Лица(u, s)+0,6*Текст(u,s) 

Где u – пользователь, для которого ищутся рекомендации, s – возможные к рекомендации пользователи. 

# Список литературы:
1. https://github.com/nlpub/pymystem3 
