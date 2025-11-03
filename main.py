import argparse

from wrapper import parse_arguments, manage_study
from docs.functions_mnist import objective, train


if __name__ == "__main__":

    args = parse_arguments()

    if args.test:
        print("Testing Train function...")
        of, seed = train(0.004, 0.15, [120,120], 1000, 1000, 0.03, 1, 0, trial_path=False, verbose=True)
        print(f"Train function terminated with value: {of:4.2f}")
        print("Everything is working")
    else:
        #Load or create a study
        study, exp_name = manage_study(args)

        #Workaround to pass two aguments to study.optimize
        func = lambda trial: objective(trial, exp_name)

        print(f'{args.n_trials} trials will be done')
        print(f'---------------------------')

        study.optimize(func, n_trials=args.n_trials)

        # Visualizza i risultati dell'ottimizzazione
        print('Miglior set di iperparametri:', study.best_params)

        print('Miglior valore di metrica:', study.best_value)