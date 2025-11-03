import os
import optuna
import argparse
from datetime import datetime

def parse_arguments():

    #Argument Parser
    parser = argparse.ArgumentParser(description='Wrapper for optuna.')
    ln = parser.add_mutually_exclusive_group(required=True)
    ln.add_argument('-l', '--load', help = 'Load the study, if more than one study is found the latest one will be used')
    ln.add_argument('-c', '--create',  help = 'Create a new study')
    
    ln.add_argument('-t', '--test', help = 'Test the Train Function', action='store_true')

    parser.add_argument('-n', '--n_trials', default=100, type=int, help = 'Define the number of trials')

    args = parser.parse_args()

    return args

def manage_study(args):

    database_path = f'sqlite:///database/trials.db'

    if args.load:
        exp_list = os.listdir('reports/')
        ls = [ x for x in exp_list if args.load in x]

        if not ls:
            print("No study was found")
            exit()
        else: 
            ls.sort(reverse=True)
            exp_name = ls[0]
            print(f'The study {exp_name} will be loaded')
            try:
                study = optuna.load_study(study_name=f'{exp_name}', storage=database_path)
            except:
                print("No study was found in db")
    else:
        counter = 0
        created = False
        while not created:
            exp_name = args.create + "-["+ datetime.now().strftime("%m-%d")+"]-" + '[' + str(counter) + ']'
            exp_path = 'reports/' + exp_name
            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
                print(f'The study {exp_name} will be created')
                created = True
            else:
                counter += 1

        try:
            study = optuna.create_study(direction='maximize', storage=database_path, study_name=f'{exp_name}')
        except:
            print("Another study with the same was found")

    return study, exp_name


def create_folder(exp_name, n):

    trial_path = 'reports/'+exp_name+'/'+str(n)

    if not os.path.exists(trial_path):
        os.makedirs(trial_path)
    else:
        print("Something went wrong")

    return trial_path

def save_report(trial_path, trial_number, epoch_report):

    with open(trial_path+"/"+str(trial_number)+"-report.txt", "a") as f:
        f.write(epoch_report + "\n")
    f.close()
