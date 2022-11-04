import os, glob, json
import argparse

def format_eval(args):
    folder = args.experiment_name
    print(folder)
    
    files = sorted(glob.glob(folder + '/*'))
    txt_files = []
    for file in files:
        if not file.endswith('bin'):
            if not file.endswith('txt'):
                if not file.endswith('json'):
                    txt_files.append(file)

    total_dic = {}
    max_overall = 0
    max_dic = {}
    print(f'Found {len(txt_files)} files')

    print(f"From {txt_files[0].split('/')[-1]} ~ to {txt_files[-1].split('/')[-1]}")

    for txt_file in txt_files:
        iter_name = txt_file.split('/')[-1].split('.')[-1]

        with open(txt_file) as f:
            lines = f.readlines()
        here = lines.index('====================== EXACT MATCHING ACCURACY =====================\n')+1
        result = [i for i in lines[here].strip().replace('exact match', '').split(' ') if i != '']

        dic = {'easy': 0, 'medium': 0, 'hard': 0, 'extra_hard': 0, 'overall': 0}
        dic_keys = ['easy', 'medium', 'hard', 'extra_hard', 'overall']
        for element, k in zip(result, dic_keys):
            dic[k] = element
        if float(dic['overall']) > float(max_overall):
            max_overall = dic['overall']
            max_dic = dic
            max_dic['iter_name'] = iter_name

        total_dic[iter_name] = dic
        print(f'{iter_name}: {dic}')
    total_dic['Best_Performance'] = max_dic
    folder = f'exp_results/{folder}'
    fileName = f'{folder}/result.json'

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(fileName, 'w') as fp:
        json.dump(total_dic, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Evaluation Folders Location etc')
    parser.add_argument('--experiment_name')

    args = parser.parse_args()

    format_eval(args)