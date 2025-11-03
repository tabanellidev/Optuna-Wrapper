# Optuna-Wrapper

The goal of this project is to enable an easier way to track experiments, reports and models saved. To use the Wrapper simply download the .zip archive, extract and edit `train` and `objective` found in `functions.py`.

## Folder Structure


### Folder Structure
The project has the following structure
```
.
├── env                           #  Python Virtual Enviroment
    └── ...
├── resources                 
    └── functions_mnist.py 
    └── requirements.txt              
├── database                      #  Log the requests into a SQLite3 Database
    └── trials.db  
├── main.py                       #  Main script to execute the bot     
├── models.py                        #  Environments Variables
└── wrapper.py
```
