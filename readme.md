# Amazone Review Rating Prediction with BERT4REC(Transformer4REC)

BERT4REC is specialized at predicting the next item that a user most likely to buy. Objective of BERT4REC is MLM(Masked Language Modeling) meaning that it can't predict numeric value.  

In this project, I modify BERT4REC to make it working as a regression model sothat the model can predict the rating of a product.


![image](https://user-images.githubusercontent.com/63226383/209775077-6d2d2df9-8696-42a7-b59e-8bf702df34d3.png)


# Quick Start
```
python main.py --data_dir "../data" --batch_size 256 --epoch_size 7
```





