## Introduction 
The example code shows how we can extract company name from copyright text of website

For example: 

        Input Text1)          = © COPYRIGHT 2019 AFR FURNITURE RENTA
        Detected Company Name =  AFR FURNITURE RENTAL
        
        Input Text2)          =  © 2019 All Rights Reserved by Abundant Life Worship Center of Whippan
        Detected Company Name =  Abundant Life Worship Center of Whippan
        
        Input Text3)          = Copyright ©Voyagers Travel
        Detected Company Name =  Voyagers Travel
        
        Input Text3)          = ABVNBYOND Inc © 2013 All Rights Reserved
        Detected Company Name = ABVNBYOND Inc

## Setup
1) > pip3 install -r requirements.txt

2) > python3 train_model.py (to train model)  

3) > python3 test_model.py (to test model)


## Reference Links:
1. [https://github.com/explosion/spaCy/blob/master/examples/training/train_ner.py](https://github.com/explosion/spaCy/blob/master/examples/training/train_ner.py)
2. [https://github.com/doccano/doccano](https://github.com/doccano/doccano)
3. [https://stackoverflow.com/questions/1707725/find-name-of-company-at-url](https://stackoverflow.com/questions/1707725/find-name-of-company-at-url)

## Developer Help:
 [https://www.linkedin.com/in/aakashkag/](https://www.linkedin.com/in/aakashkag/)

